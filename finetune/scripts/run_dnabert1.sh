#!/bin/bash

# This is your argument
data_path=$1
kmer=$2

echo "The provided kmer is: $kmer, data_path is $data_path"

# sh scripts/run_dna1.sh 3 ; sh scripts/run_dna1.sh 4 ; sh scripts/run_dna1.sh 5 ; sh scripts/run_dna1.sh 6

for seed in 42
do
    for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    do
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/EMP/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_EMP_${data}_seed${seed} \
            --model_max_length 512 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
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
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/dnabert1_${kmer} \
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
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 80 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
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
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 310 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/dnabert1_${kmer} \
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
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/prom/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_prom_${data}_seed${seed} \
            --model_max_length 310 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
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
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/splice/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_splice_${data}_seed${seed} \
            --model_max_length 410 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
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
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/virus/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_virus_${data}_seed${seed} \
            --model_max_length 1024 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --learning_rate 3e-5 \
            --num_train_epochs 9 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/mouse/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_mouse_${data}_seed${seed} \
            --model_max_length 110 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 5 \
            --max_steps 1000 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 30 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done


    for data in 0 1 2 3 4
    do 
        python train.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/tf/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_tf_${data}_seed${seed} \
            --model_max_length 110 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/dnabert1_${kmer} \
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