# debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"

# llama-7b-2k-16k-linear / gov_report dataset / truncation for ppl, no sliding window

prefix=path_to_your_model_and_data

for factor in 8
do
    python ${debug_mode} src/eval_ppl.py \
        --path_to_ckp ${prefix}/skipos/results/2k-$((factor*2))k-linear/checkpoint-1000 \
        --model_name llama-7b-2k-$((factor*2))k-linear \
        --rope_scaling_type linear \
        --rope_scaling_factor ${factor} \
        --model_max_position_embeddings 2048 \
        --max_input_tokens 32768 \
        --min_input_tokens 32768 \
        --window_length_list 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384  \
        --truncate \
        --batch_size 1 \
        --dataset_name scrolls-gov_report \
        --path_to_dataset ${prefix}/data/scrolls/gov_report/test_long.jsonl
done

# llama-7b-2k-96k-yarn / pg19 dataset / sliding window for ppl 
for factor in 48
do
    python ${debug_mode} src/eval_ppl.py \
        --path_to_ckp ${prefix}/skipos/results/2k-$((factor*2))k-yarn/checkpoint-900 \
        --model_name llama-7b-2k-$((factor*2))k-yarn \
        --rope_scaling_type yarn \
        --rope_scaling_factor ${factor} \
        --model_max_position_embeddings 2048 \
        --max_input_tokens 131072 \
        --min_input_tokens 131072 \
        --sliding_window_step 16384 \
        --window_length_list 32768 65536 98304 \
        --batch_size 1 \
        --eval_nums 20 \
        --dataset_name pile-pg19 \
        --path_to_dataset ${prefix}/data/pile/test_pg19.jsonl
done

