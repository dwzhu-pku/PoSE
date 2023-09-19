# debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"

prefix=path_to_your_model_and_data


python ${debug_mode} src/eval_passkey.py \
    --path_to_ckp ${prefix}/model/llama-7b \
    --model_name llama-7b \
    --rope_scaling_type linear \
    --rope_scaling_factor ${factor}

python ${debug_mode} src/eval_passkey.py \
    --path_to_ckp ${prefix}/skipos/results/2k-16k-linear/checkpoint-1000 \
    --model_name llama-7b-2k-16k-linear \
    --rope_scaling_type linear \
    --rope_scaling_factor 8


python ${debug_mode} src/eval_passkey.py \
    --path_to_ckp ${prefix}/skipos/results/2k-32k-linear/checkpoint-1000 \
    --model_name llama-7b-2k-32k-linear \
    --rope_scaling_type linear \
    --rope_scaling_factor 16

