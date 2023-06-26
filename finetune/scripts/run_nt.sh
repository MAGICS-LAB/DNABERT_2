#!/bin/bash

data_path=$1
m=$2


if [ "$m" -eq 0 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-1000g
    run_name=NT_500_1000g
elif [ "$m" -eq 1 ]; then
    model=InstaDeepAI/nucleotide-transformer-500m-human-ref
    run_name=NT_500_human
elif [ "$m" -eq 2 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-1000g
    run_name=NT_2500_1000g
elif [ "$m" -eq 3 ]; then
    model=InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    run_name=NT_2500_multi
else
    echo "Wrong argument"
    exit 1
fi
echo "Use: $model"


for seed in 42
do
    for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/EMP/$data \
            --kmer -1 \
            --run_name ${run_name}_EMP_${data}_seed${seed} \
            --model_max_length 100 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done



    for data in prom_core_all prom_core_notata
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${run_name}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in prom_core_tata
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${run_name}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${run_name}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in prom_300_tata
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer -1 \
            --run_name ${run_name}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in reconstructed
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/splice/$data \
            --kmer -1 \
            --run_name ${run_name}_splice_${data}_seed${seed} \
            --model_max_length 80 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done



    for data in covid
    do
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/virus/$data \
            --kmer -1 \
            --run_name ${run_name}_virus_${data}_seed${seed} \
            --model_max_length 256 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 10000 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 200 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done

    for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/mouse/$data \
            --kmer -1 \
            --run_name ${run_name}_mouse_${data}_seed${seed} \
            --model_max_length 30 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 5 \
            --max_steps 1000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 100 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path ${model} \
            --data_path  ${data_path}/GUE/tf/$data \
            --kmer -1 \
            --run_name ${run_name}_tf_${data}_seed${seed} \
            --model_max_length 30 \
            --use_lora \
            --lora_target_modules 'query,value,key,dense' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --lora_alpha 16 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/nt_${run_name} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done