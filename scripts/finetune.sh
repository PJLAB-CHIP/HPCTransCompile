python finetune.py \
    --model_name_or_path /code/model/CodeLlama-13b-hf \
    --output_dir output/hpc_v3.0_topi_without_ir_by_name_simplify \
    --dataset_name_or_path /code/LLM4HPCTransCompile/preprocess/data/hpc_v3.0_topi_without_ir_by_name_simplify/hpc_train_v3.0_topi_without_ir_by_name_simplify.json \
    --dataset_format input-output \
    --use_wandb True \
    --wandb_run_name hpc_v3.0_topi_without_ir_by_name_simplify_3_15 \
    --device_map auto \
    --full_finetune True \
    --source_max_len 2048 \
    --target_max_len 4096 \
    --use_lora True \
    

#### Llama-2-7b-hf ####
# python finetune.py \
#     --model_name_or_path /code/model/Llama-2-7b-hf \
#     --output_dir output/hpc_v3.0_topi_without_ir_by_name_simplify \
#     --dataset_name_or_path /code/LLM4HPCTransCompile/preprocess/data/hpc_v3.0_topi_without_ir_by_name_simplify/hpc_train_v3.0_topi_without_ir_by_name_simplify.json \
#     --dataset_format input-output \
#     --use_wandb True \
#     --wandb_run_name hpc_v3.0_topi_without_ir_by_name_simplify \
#     --device_map auto \
#     --full_finetune True \
#     --batch_size 1 \
#     --source_max_len 512 \
#     --target_max_len 512 \


#### test ####
# python finetune.py \
#     --model_name_or_path /code/model/CodeLlama-13b-hf \
#     --output_dir output/hpc_test \
#     --dataset_name_or_path /code/LLM4HPCTransCompile/preprocess/data/hpc_test/hpc_train_test.json \
#     --dataset_format input-output \
#     --use_wandb True \
#     --wandb_run_name hpc_test \
#     --device_map cuda
