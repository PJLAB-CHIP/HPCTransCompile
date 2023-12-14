# python qlora.py \
#     --dataset ../preprocess/data/hpc_train_v1.0.json \
#     --model_name_or_path ../../model/CodeLlama-13b-hf \
#     --learning_rate 0.0001 \
#     --num_train_epochs 10 \
#     --bits 8 \
#     --dataset_format alpaca \
#     --output_dir ./output/hpc_v1.1 \
#     --per_device_train_batch_size 1 \

python qlora.py \
    --dataset ../preprocess/data/hpc_v1.5_with_ir/hpc_train_v1.5_with_ir.json \
    --model_name_or_path ../../model/CodeLlama-13b-hf \
    --learning_rate 0.0001 \
    --num_train_epochs 10 \
    --bits 8 \
    --dataset_format alpaca \
    --output_dir ./output/hpc_v1.5_with_ir \
    --per_device_train_batch_size 5 \
    --save_steps 100