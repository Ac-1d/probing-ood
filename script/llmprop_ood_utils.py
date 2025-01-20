# Import packages
import glob
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import re
import json
import glob
import tarfile
import datetime
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import SGD

import matplotlib.pyplot as plt

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

# for metrics
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import roc_auc_score

np.random.seed(42)
torch.manual_seed(42)

def tokenize(tokenizer, X_df, y_df, max_length, pooling='cls'):
    """
    1. Takes in the the list of input sequences and return 
    the input_ids and attention masks of the tokenized sequences
    2. max_length = the max length of each input sequence 
    """
    if pooling == 'cls':
        encoded_corpus = tokenizer(text=["[CLS] " + str(descr) for descr in X_df.tolist()],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True)
    elif pooling == 'mean':
        encoded_corpus = tokenizer(text=X_df.tolist(),
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True) 
    input_ids = encoded_corpus['input_ids']
    attention_masks = encoded_corpus['attention_mask']

    return input_ids, attention_masks

def create_dataloaders(tokenizer, X_df, y_df, max_length, batch_size, property_value="band_gap", pooling='cls', normalize=False, normalizer='z_norm'):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, X_df, y_df, max_length, pooling=pooling)
    labels = y_df.to_numpy()

    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    labels_tensor = torch.tensor(labels)

    if normalize:
        if normalizer == 'z_norm':
            normalized_labels = z_normalizer(labels_tensor)
        elif normalizer == 'mm_norm':
            normalized_labels = min_max_scaling(labels_tensor)
        elif normalizer == 'ls_norm':
            normalized_labels = log_scaling(labels_tensor)
        elif normalizer == 'no_norm':
            normalized_labels = labels_tensor

        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, normalized_labels)
    else:
        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Set the shuffle to False for now since the labels are continues values check later if this may affect the result

    return dataloader

class T5Predictor(nn.Module):
    def __init__(
        self, 
        base_model, 
        base_model_output_size,  
        n_classes=1, 
        drop_rate=0.5, 
        freeze_base_model=False, 
        bidirectional=True, 
        pooling='cls'
    ):
        super(T5Predictor, self).__init__()
        D_in, D_out = base_model_output_size, n_classes
        self.model = base_model
        self.dropout = nn.Dropout(drop_rate)
        self.pooling = pooling

        # instantiate a linear layer
        self.linear_layer = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )

    def forward(self, input_ids, attention_masks):
        hidden_states = self.model(input_ids, attention_masks)

        last_hidden_state = hidden_states.last_hidden_state # [batch_size, input_length, D_in]

        if self.pooling == 'cls':
            input_embedding = last_hidden_state[:,0,:] # [batch_size, D_in] -- [CLS] pooling
        elif self.pooling == 'mean':
            input_embedding = last_hidden_state.mean(dim=1) # [batch_size, D_in] -- mean pooling
        
        outputs = self.linear_layer(input_embedding) # [batch_size, D_out]

        return input_embedding, outputs

def writeToJSON(data, where_to_save):
    """
    data: a dictionary that contains data to save
    where_to_save: the name of the file to write on
    """
    with open(where_to_save, "w", encoding="utf8") as outfile:
        json.dump(data, outfile)

def readJSON(input_file):
    """
    1. arguments
        input_file: a json file to read
    2. output
        a json objet in a form of a dictionary
    """
    with open(input_file, "r", encoding="utf-8", errors='ignore') as infile:
        json_object = json.load(infile, strict=False)
    return json_object

def writeTEXT(data, where_to_save):
    with open(where_to_save, "w", encoding="utf-8") as outfile:
        for d in data:
            outfile.write(str(d))
            outfile.write("\n")

def readTEXT_to_LIST(input_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = []
        for line in infile:
            data.append(line)
    return data

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)

def time_format(total_time):
    """
    Change the from seconds to hh:mm:ss
    """
    total_time_rounded = int(round((total_time)))
    total_time_final = str(datetime.timedelta(seconds=total_time_rounded))
    return total_time_final

def z_normalizer(labels):
    """ Implement a z-score normalization technique"""
    labels_mean = torch.mean(labels)
    labels_std = torch.std(labels)

    scaled_labels = (labels - labels_mean) / labels_std

    return scaled_labels

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

def min_max_scaling(labels):
    """ Implement a min-max normalization technique"""
    min_val = torch.min(labels)
    max_val = torch.max(labels)
    diff = max_val - min_val
    scaled_labels = (labels - min_val) / diff
    return scaled_labels

def mm_denormalize(scaled_labels, min_val, max_val):
    diff = max_val - min_val
    denorm_labels = (scaled_labels * diff) + min_val
    return denorm_labels

def log_scaling(labels):
    """ Implement log-scaling normalization technique"""
    scaled_labels = torch.log1p(labels)
    return scaled_labels

def ls_denormalize(scaled_labels):
    denorm_labels = torch.expm1(scaled_labels)
    return denorm_labels

def compressCheckpointsWithTar(filename):
    filename_for_tar = filename[0:-3]
    tar = tarfile.open(f"{filename_for_tar}.tar.gz", "w:gz")
    tar.add(filename)
    tar.close()

def decompressTarCheckpoints(tar_filename):
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()

def replace_bond_lengths_with_num(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "[NUM]", sentence) # Regex pattern to match bond lengths and units
    return sentence.strip()

def replace_bond_angles_with_ang(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*°", "[ANG]", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*degrees", "[ANG]", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def replace_bond_lengths_and_angles_with_num_and_ang(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "[NUM]", sentence) # Regex pattern to match bond lengths and units
    sentence = re.sub(r"\d+(\.\d+)?(?:-\d+(\.\d+)?)?\s*°", "[ANG]", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:-\d+(\.\d+)?)?\s*degrees", "[ANG]", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def get_cleaned_stopwords():
    # from https://github.com/igorbrigadir/stopwords
    stopword_files = glob.glob("../stopwords/en/*.txt")
    num_str = {'one','two','three','four','five','six','seven','eight','nine'}

    all_stopwords_list = set()

    for file_path in stopword_files:
        all_stopwords_list |= set(readTEXT_to_LIST(file_path))

    cleaned_list_for_mat = {wrd.replace("\n", "").strip() for wrd in all_stopwords_list} - {wrd for wrd in all_stopwords_list if wrd.isdigit()} - num_str
    
    return cleaned_list_for_mat

def remove_mat_stopwords(sentence):
    stopwords_list = get_cleaned_stopwords()
    words = sentence.split()
    words_lower = sentence.lower().split()
    sentence = ' '.join([words[i] for i in range(len(words)) if words_lower[i] not in stopwords_list])
    return sentence.strip()

def get_sequence_len_stats(df, tokenizer, max_len):
    training_on = sum(1 for sent in df.apply(tokenizer.tokenize) if len(sent) <= max_len)
    return (training_on/len(df))*100

def get_roc_score(predictions, targets):
    roc_fn = BinaryAUROC(threshold=None)
    x = torch.tensor(targets)
    y = torch.tensor(predictions)
    y = torch.round(torch.sigmoid(y))
    roc_score = roc_fn(y, x)
    return roc_score

def train_and_predict(
    model, 
    optimizer, 
    scheduler, 
    bce_loss_function, 
    mae_loss_function,
    epochs, 
    train_dataloader, 
    valid_dataloader,
    test_dataloader, 
    device,
    y_train_mean,
    y_train_std,  
    normalizer="z_norm",
    task_name="regression"
):
    
    training_starting_time = time.time()
    validation_predictions = {}
    best_y_pred = []
    
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    
    best_loss = 1e10 # Set the best loss variable which record the best loss for each epoch
    best_roc = 0.0

    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} =========")

        epoch_starting_time = time.time() 

        total_training_loss = 0
        total_training_mae_loss = 0
        total_training_normalized_mae_loss = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            print(f"Step {step+1}/{len(train_dataloader)}")

            batch_inputs, batch_masks, batch_labels, batch_norm_labels = tuple(b.to(device) for b in batch)

            _, predictions = model(batch_inputs, batch_masks)

            if task_name == 'classification':
                loss = bce_loss_function(predictions.squeeze(), batch_labels.squeeze())
            
            elif task_name == 'regression':
                loss = mae_loss_function(predictions.squeeze(), batch_norm_labels.squeeze())
                
                if normalizer == 'z_norm':
                    predictions_denorm = z_denormalize(predictions, y_train_mean, y_train_std)

                elif normalizer == 'mm_norm':
                    predictions_denorm = mm_denormalize(predictions, y_train_min, y_train_max)

                elif normalizer == 'ls_norm':
                    predictions_denorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    loss = mae_loss_function(predictions.squeeze(), batch_labels.squeeze())
                    predictions_denorm = predictions

                mae_loss = mae_loss_function(predictions_denorm.squeeze(), batch_labels.squeeze()) 

            # total training loss on actual output
            if task_name == "classification":
                total_training_loss += loss.item()
            
            elif task_name == "regression":
                total_training_loss += mae_loss.item()

            # back propagate
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # average training loss on actual output
        average_training_loss = total_training_loss/len(train_dataloader) 
        
        train_loss_list.append(average_training_loss)
        
        epoch_ending_time = time.time()
        training_time = time_format(epoch_ending_time - epoch_starting_time)

        print(f"Average training loss = {average_training_loss}")
        print(f"Training for this epoch took {training_time}")

        # Validation
        print("")
        print("Running Validation ....")

        valid_start_time = time.time()

        model.eval()

        total_eval_mae_loss = 0
        predictions_list = []
        targets_list = []

        for step, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)

            with torch.no_grad():
                _, predictions = model(batch_inputs, batch_masks)

                if task_name == "classification":
                    predictions_denorm = predictions

                elif task_name == "regression":
                    if normalizer == 'z_norm':
                        predictions_denorm = z_denormalize(predictions, y_train_mean, y_train_std)

                    elif normalizer == 'mm_norm':
                        predictions_denorm = mm_denormalize(predictions, y_train_min, y_train_max)

                    elif normalizer == 'ls_norm':
                        predictions_denorm = ls_denormalize(predictions)

                    elif normalizer == 'no_norm':
                        predictions_denorm = predictions

            predictions = predictions_denorm.detach().cpu().numpy()
            targets = batch_labels.detach().cpu().numpy()

            for i in range(len(predictions)):
                predictions_list.append(predictions[i][0])
                targets_list.append(targets[i])
        
        valid_ending_time = time.time()
        validation_time = time_format(valid_ending_time-valid_start_time)

        # save model checkpoint and the statistics of the epoch where the model performs the best
        if task_name == "classification":
            valid_performance = get_roc_score(predictions_list, targets_list)
            
            if valid_performance >= best_roc:
                best_roc = valid_performance
                best_epoch = epoch+1
                best_y_pred = predictions_list

            else:
                best_roc = best_roc

            print(f"Validation roc score = {valid_performance}")

        elif task_name == "regression":
            predictions_tensor = torch.tensor(predictions_list)
            targets_tensor = torch.tensor(targets_list)
            valid_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
        
            if valid_performance <= best_loss:
                best_loss = valid_performance
                best_epoch = epoch+1
                best_y_pred = predictions_list

            else:
                best_loss = best_loss
            
            print(f"Validation mae error = {valid_performance}")
        
        valid_loss_list.append(valid_performance)
        
        print(f"validation took {validation_time}")

        # Validation
        print("")
        print("Running Testing ....")

        test_start_time = time.time()

        model.eval()

        total_test_mae_loss = 0
        predictions_list_test = []
        targets_list = []

        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)

            with torch.no_grad():
                _, predictions = model(batch_inputs, batch_masks)

                if task_name == "classification":
                    predictions_denorm = predictions

                elif task_name == "regression":
                    if normalizer == 'z_norm':
                        predictions_denorm = z_denormalize(predictions, y_train_mean, y_train_std)

                    elif normalizer == 'mm_norm':
                        predictions_denorm = mm_denormalize(predictions, y_train_min, y_train_max)

                    elif normalizer == 'ls_norm':
                        predictions_denorm = ls_denormalize(predictions)

                    elif normalizer == 'no_norm':
                        predictions_denorm = predictions

            predictions = predictions_denorm.detach().cpu().numpy()
            targets = batch_labels.detach().cpu().numpy()

            for i in range(len(predictions)):
                predictions_list_test.append(predictions[i][0])
                targets_list.append(targets[i])
        
        test_ending_time = time.time()
        test_time = time_format(test_ending_time-test_start_time)

        # save model checkpoint and the statistics of the epoch where the model performs the best
        if task_name == "classification":
            test_performance = get_roc_score(predictions_list_test, targets_list)
            
            if valid_performance >= best_roc:
                best_roc = valid_performance
                best_epoch = epoch+1
                best_y_pred_test = predictions_list_test

            else:
                best_roc = best_roc

            print(f"Test roc score = {test_performance}")

        elif task_name == "regression":
            predictions_tensor = torch.tensor(predictions_list_test)
            targets_tensor = torch.tensor(targets_list)
            test_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
        
            if valid_performance <= best_loss:
                best_loss = valid_performance
                best_epoch = epoch+1
                best_y_pred_test = predictions_list_test

            else:
                best_loss = best_loss
            
            print(f"Test mae error = {test_performance}")
            
        test_loss_list.append(test_performance)
        
        print(f"test took {validation_time}")
    
    train_ending_time = time.time()
    total_training_time = train_ending_time-training_starting_time
    
    learning_curve = pd.DataFrame({
        'epoch':range(epochs),
        'train_mae_error':train_loss_list, 
        'valid_mae_error':valid_loss_list, 
        'test_mae_error':test_loss_list
    })

    print("\n========== Training complete ========")
    print(f"Training LLM_Prop took {time_format(total_training_time)}")

    if task_name == "classification":
        print(f"The lowest roc score is {best_roc} at {best_epoch}th epoch \n")

    elif task_name == "regression":
        print(f"The lowest mae error is {best_loss} at {best_epoch}th epoch \n")
    
    return best_y_pred_test, learning_curve

def train_and_predict_main(
    X_train, 
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test
):
    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Training and testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")
    
    ###################################################################
    # @change parameters as you would like here
    # set parameters
    batch_size = 64
    max_length = 1500
    learning_rate = 0.001
    drop_rate = 0.5
    epochs = 50
    warmup_steps = 50000
    preprocessing_strategy = "no_stopwords_and_lengths_and_angles_replaced"
    tokenizer_name = 'modified' 
    pooling = 'cls'
    scheduler_type = 'onecycle'
    normalizer_type = 'z_norm'
    optimizer_type = 'adamw'
    #######################################################################

    # check property type to determine the task name (whether it is regression or classification)
    if y_train.dtype == 'bool':
        task_name = 'classification'

        #converting True->1.0 and False->0.0
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)  
    else:
        task_name = 'regression'
    
    y_train_array = np.array(y_train)
    y_train_mean = torch.mean(torch.tensor(y_train_array))
    y_train_std = torch.std(torch.tensor(y_train_array))
    y_train_min = torch.min(torch.tensor(y_train_array))
    y_train_max = torch.max(torch.tensor(y_train_array))

    if preprocessing_strategy == "none":
        X_train = X_train
        X_valid = X_valid
        X_test = X_test

    elif preprocessing_strategy == "bond_lengths_replaced_with_num":
        X_train = X_train.apply(replace_bond_lengths_with_num)
        X_valid = X_valid.apply(replace_bond_lengths_with_num)
        X_test = X_test.apply(replace_bond_lengths_with_num)
        print(X_train[0])
        print('-'*50)
        print(X_train[3])

    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        X_train = X_train.apply(replace_bond_angles_with_ang)
        X_valid = X_valid.apply(replace_bond_angles_with_ang)
        X_test = X_test.apply(replace_bond_angles_with_ang) 
        print(X_train[0])
        print('-'*50)
        print(X_train[3])

    elif preprocessing_strategy == "no_stopwords":
        stopwords = get_cleaned_stopwords()
        X_train = X_train.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        X_valid = X_valid.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        X_test = X_test.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        print(X_train[0])
        print('-'*50)
        print(X_test[1])

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        # stopwords = get_cleaned_stopwords()
        X_train = X_train.apply(replace_bond_lengths_with_num)
        X_train = X_train.apply(replace_bond_angles_with_ang)
        X_train = X_train.apply(remove_mat_stopwords) 
        X_valid = X_valid.apply(replace_bond_lengths_with_num)
        X_valid = X_valid.apply(replace_bond_angles_with_ang)
        X_valid = X_valid.apply(remove_mat_stopwords)
        X_test = X_test.apply(replace_bond_lengths_with_num)
        X_test = X_test.apply(replace_bond_angles_with_ang)
        X_test = X_test.apply(remove_mat_stopwords)
        print(X_train[0])
        print('-'*50)
        print(X_test[1])

    # define loss functions
    mae_loss_function = nn.L1Loss()
    bce_loss_function = nn.BCEWithLogitsLoss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights

    # define the tokenizer
    if tokenizer_name == 't5_tokenizer': 
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    elif tokenizer_name == 'modified':
        tokenizer = AutoTokenizer.from_pretrained("tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge_v1")

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])

    if preprocessing_strategy == "bond_lengths_replaced_with_num":
        tokenizer.add_tokens(["[NUM]"]) # special token to replace bond lengths
    
    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        tokenizer.add_tokens(["[ANG]"]) # special token to replace bond angles

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        tokenizer.add_tokens(["[NUM]"])
        tokenizer.add_tokens(["[ANG]"]) 
    
    print('-'*50)
    print(f"train data = {len(X_train)} samples")
    print(f"valid data = {len(X_valid)} samples")
    print(f"test data = {len(X_test)} samples")
    print('-'*50)
    print(f"training on {get_sequence_len_stats(X_train, tokenizer, max_length)}% samples with whole sequence")
    print(f"validating on {get_sequence_len_stats(X_valid, tokenizer, max_length)}% samples with whole sequence")
    print(f"testing on {get_sequence_len_stats(X_test, tokenizer, max_length)}% samples with whole sequence")
    print('-'*50)

    print("labels statistics on training set:")
    print("Mean:", y_train_mean)
    print("Standard deviation:", y_train_std)
    print("Max:", y_train_max)
    print("Min:", y_train_min)
    print("-"*50)

    # define the model
    base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
    base_model_output_size = 512

    # freeze the pre-trained LM's parameters
    if freeze:
        for param in base_model.parameters():
            param.requires_grad = False

    # resizing the model input embeddings matrix to adapt to newly added tokens by the new tokenizer
    # this is to avoid the "RuntimeError: CUDA error: device-side assert triggered" error
    base_model.resize_token_embeddings(len(tokenizer))

    # instantiate the model
    model = T5Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling)

    device_ids = [d for d in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model.to(device)

    # print the model parameters
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters = {model_trainable_params}")

    # create dataloaders
    train_dataloader = create_dataloaders(
        tokenizer, 
        X_train,
        y_train, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling, 
        normalize=True, 
        normalizer=normalizer_type
    )

    valid_dataloader = create_dataloaders(
        tokenizer, 
        X_valid,
        y_valid, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling
    )

    test_dataloader = create_dataloaders(
        tokenizer, 
        X_test,
        y_test, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling
    )

    # define the optimizer
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr = learning_rate
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=learn_rate
        )

    # set up the scheduler
    total_training_steps = len(train_dataloader) * epochs 
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup( #get_linear_schedule_with_warmup
            optimizer,
            num_warmup_steps= warmup_steps, #steps_ratio*total_training_steps,
            num_training_steps=total_training_steps 
        )
    
    # from <https://github.com/usnistgov/alignn/blob/main/alignn/train.py>
    elif scheduler_type == 'onecycle': 
        steps_per_epoch = len(train_dataloader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    
    elif scheduler_type == 'step':
         # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=warmup_steps
        )
    
    elif scheduler_type == 'lambda':
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    
    print("======= Training ... ========")
    y_pred, learning_curve = train_and_predict(
        model, 
        optimizer, 
        scheduler, 
        bce_loss_function, 
        mae_loss_function, 
        epochs, 
        train_dataloader,
        valid_dataloader, 
        test_dataloader, 
        device,
        y_train_mean,
        y_train_std, 
        normalizer=normalizer_type,
        task_name=task_name
    )
    
    print('The total test sample =', len(y_pred))
    print("======= Training finished ========")

    return y_pred, learning_curve
