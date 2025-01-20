**Under construction**

The environment.yml file specifies the conda virtual environment. To create the environment, run the following command:
```
conda env create -f environment.yml # if using Linux
```
or 
```
conda env create -f environment_no_builds.yml # for more generic installation
```

sub_job.bash is the job sumission script to the high-performing computing clusters for all tasks. 

Python files:
    - leave_one_group_out.py: run the standard OOD tests. This is the main file to run the OOD evaluation. Examples to run the evaluation can be found in sub_job.bash. 
         Basically it can be run with the following syntax: `python leave_one_group_out.py --dataset $dataset --target $target --modelname $modelname --group_label $group_label --group_value_list="$group_value_list" `
    - get_learning_curve.py: this is the file to record and get the loss vs. number of epoch curves. See run_get_learning_curve.bash for examples to run the evaluation.
    - get_ood_scores_vs_size.py: this is the file to run the OOD evaluation to get the loss vs. number of training data curves. See run_get_ood_scores_vs_size.bash for examples to run the evaluation.
    - track_model_update.py: this is the file to perform SHAP analysis on the leave-one-element-out tasks using the XGBoost model. This can be run independently 
    - dim_reduction.py: this is the file used to generate the UMAP plots, after running the 
    - Other python files store the functions used in the above files. 


