#!/bin/bash



modelname='alignn1000'
for group_value in H;do
    python get_learning_curve.py \
        --modelname $modelname \
        --epochs 1000 \
        --dataset jarvis22 --target e_form \
        --group_label elements --group_value $group_value \
         && echo "@@ sucess: $group_value" 
done

modelname='gmp'
for group_value in H;do
    python get_learning_curve.py \
        --modelname $modelname \
        --epochs 1000 \
        --dataset jarvis22 --target e_form \
        --group_label elements --group_value $group_value \
         && echo "@@ sucess: $group_value" 
done





# dataset='mp21'
# for modelname in xgb gmp alignn200;do
#     for group_value in H F O Fe Bi Li N C;do 
#         python get_learning_curve.py \
#             --modelname $modelname \
#             --dataset $dataset --target e_form \
#             --group_label elements --group_value $group_value && echo "@@ sucess: $modelname $group_value"
#     done 
# done



