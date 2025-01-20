#!/bin/bash

set -eu 
group_value_list="F H O" # C N O F Fe K Ba"
#group_value_list='C N O F Fe Cu Ba H'



for dataset in oqmd21;do
    for target in e_form;do
        for modelname in xgb;do
            for group_label in elements;do
#                for ood_train_size in 50 250 1250;do
                #  for ood_train_size in 50 250 500 1000 2000;do
                for ood_train_size in 10 100 1000;do
                    python get_ood_scores_vs_size.py \
                        --dataset $dataset \
                        --target $target \
                        --modelname $modelname \
                        --group_label $group_label \
                        --group_value_list="$group_value_list" \
                        --ood_train_size="$ood_train_size" \

                done
            done
        done
    done
done

