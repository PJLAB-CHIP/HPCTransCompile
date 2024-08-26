export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
python finetune.py \
    --model_name_or_path /code/LLM4HPCTransCompile/model/starcoder2-7b \
    --output_dir output/hpc_v3.0_topi_without_ir_by_name_simplify \
    --dataset_name_or_path /code/LLM4HPCTransCompile/dataset/dataset_convert_train.json\
    --dataset_format input-output \
    --use_wandb True \
    --wandb_run_name hpc_v3.0_topi_without_ir_by_name_simplify_3_15 \
    --device_map auto \
    --full_finetune False \
    --source_max_len 2048 \
    --target_max_len 4096 \
    --use_lora True \