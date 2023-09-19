# debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"
ZERO_SHOT="--tasks boolq,piqa,winogrande,truthfulqa_mc"
HELLASWAG="--tasks hellaswag --num_fewshot 10"
ARC="--tasks arc_challenge --num_fewshot 25"

path_to_lm_eval="helper/lm-evaluation-harness"
path_to_model="your_path_to_model"

python ${debug_mode} ${path_to_lm_eval}/main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=${path_to_model},use_accelerate=True,dtype=float16" \
    ${ZERO_SHOT} \
    --device cuda:0

python ${debug_mode} ${path_to_lm_eval}/main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=${path_to_model},use_accelerate=True,dtype=float16" \
    ${HELLASWAG} \
    --device cuda:0

python ${debug_mode} ${path_to_lm_eval}/main.py \
    --model hf-causal-experimental \
    --model_args "pretrained=${path_to_model},use_accelerate=True,dtype=float16" \
    ${ARC} \
    --device cuda:0   




