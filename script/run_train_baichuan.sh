flag=$1
factor=$2
rope_type=$3

prefix=path_to_your_model_and_data

debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"
# python -m torch.distributed.run --nproc_per_node=1 ${debug_mode} src/train_baichuan.py \
deepspeed src/train_baichuan.py \
    --model_name_or_path $prefix/model/Baichuan2-7B-Base \
    --train_data_path $prefix/data/pile/train/00_long_10w.jsonl \
    --valid_data_path $prefix/data/pile/val_long.jsonl \
    --test_data_path $prefix/data/pile/test_pg19.jsonl \
    --output_dir $prefix/skipos/results/baichuan2/4k-$((factor*4))k-${rope_type} \
    --max_steps 1000 \
    --model_max_position_embeddings 4096 \
    --rope_scaling_type ${rope_type} \
    --rope_scaling_factor $factor \
    --inference_length 16384 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --fp16 True \
    --deepspeed src/configs/deepspeed_config.json \