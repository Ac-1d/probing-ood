#!/bin/bash

set -eu

declare -A group
declare -a group_label_ordered

# list of elements, excluding noble gases
group['elements']='H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts'
group_label_ordered+=('elements')
# list of groups in the periodic table, excluding noble gases
group['group']=$(seq 1 17)
group_label_ordered+=('group')
# list of periods in the periodic table, excluding the first row (because it contains H if we exclude He)
group['period']=$(seq 2 7)
group_label_ordered+=('period')
# list of space_group_number
group['space_group_number']=$(seq 1 230)
group_label_ordered+=('space_group_number')
# list of point_group (Hermann–Mauguin notation)
group['point_group']='1 -1 2 m 2/m 222 mm2 mmm 4 -4 4/m 422 4mm -42m 4/mmm 3 -3 32 3m -3m 6 -6 6/m 622 6mm -6m2 6/mmm 23 m-3 432 -43m m-3m'
group_label_ordered+=('point_group')
# list of crystal_system
group['crystal_system']='triclinic monoclinic orthorhombic tetragonal trigonal hexagonal cubic'
group_label_ordered+=('crystal_system')
# list of greater_than_nelements
group['greater_than_nelements']='2 3 4 5'
group_label_ordered+=('greater_than_nelements')
# list of le_nelements
group['le_nelements']='2 3 4'
group_label_ordered+=('le_nelements')

mkdir -p output

# loop over all the models: 
for modelname in xgb rf alignn25 gmp llm;do 

    # dataset
    for dataset in jarvis22 mp21 oqmd21;do 

        for target in e_form;do 

            # loop over all the group labels
           for group_label in "${group_label_ordered[@]}";do

                # Run in parallel: we need to create a new job for each group_value
                for group_value in ${group[$group_label]};do  

                    # remove '/' in group_value and save the result to group_value_
                    # This is specificall for point_group, because some of them contain '/'
                    group_value_=$(echo $group_value | sed 's/\///g')

                    # check if csv results exist, if yes, then skip
                    base_csv="leave_one_group_out/$target/${dataset}_rf_leave_one_${group_label}_out_pred_${group_value_}.csv"
                    csv="leave_one_group_out/$target/${dataset}_${modelname}_leave_one_${group_label}_out_pred_${group_value_}.csv"

                    # skip if csv exists
                    if [[ -f $csv ]];then
                        continue
                    else
                        # skip if base_csv does not exist (which occurs when the number of test data is too few). 
                        if [[ ! -f $base_csv ]];then
                            continue
                        else
                            echo "$csv does not exist, continue to submit job"

                            # Uncomment the following to have a dry run of submitting jobs
                            # continue
                        fi
                    fi


                    group_value_list=$group_value # Here we provide a single value group_value to group_value_list, because we don't want to run in serial.
                    jobname="$dataset.$target.$group_label.$group_value_"

                    # nhours = 3 if dataset = jarvis22, nhours = 6 if dataset = mp21, nhours = 48 if dataset = oqmd21
                    if [[ $dataset == 'jarvis22' ]];then
                        nhours=3
                        mem="31G"
                    elif [[ $dataset == 'mp21' ]];then
                        nhours=6
                        mem="63G"
                        if [[ $target == 'bulk_modulus' ]];then 
                            nhours=3
                            mem="63G"
                        fi
                    elif [[ $dataset == 'oqmd21' ]];then
                        nhours=48
                    fi


cat > job << EOF
#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --output=output/out.$jobname
#SBATCH --error=output/out.$jobname
#SBATCH --time=$nhours:00:00
#SBATCH -J $jobname
#SBATCH --cpus-per-task 6
#SBATCH --gpus-per-node=1
#SBATCH --mem=$mem

module load python/3.10 scipy-stack arrow gcc/9.3.0
source /home/likangmi/projects/def-j3goals/likangmi/myenv/hpc_ml/bin/activate
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/python/site-packages:/home/likangmi/bin/ml:/cvmfs/soft.computecanada.ca/custom/python/site-packages:/home/likangmi/bin/ml:/home/likangmi/bin/ml:/home/likangmi/bin/ml

# Since each run may take some time, better to wrap the below command 
# in a bash script and submit it as a job, so that we can run in parallel.
python leave_one_group_out.py \
    --dataset $dataset \
    --target $target \
    --modelname $modelname \
    --group_label $group_label \
    --group_value_list="$group_value_list" # use the equal sign to avoid argument string that starts with '-' to be treated as an option

EOF

    
                    sbatch job

                done     
            done
        done 
    done 
done
