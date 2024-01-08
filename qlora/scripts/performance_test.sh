# train_data_version 和 ckpt 指定了要使用的model
# test_data_version 指定了测试的数据集
# python performance_test.py \
#     --model_name CodeLlama-13b-hf \
#     --train_data_version v2.0_nn_topi_alpaca \
#     --test_data_version v2.0_nn_topi_alpaca \
#     --ckpt 100 \
#     --device 1 \
#     --use_lora_model True \
#     --bits 8

python performance_test.py \
    --model_name CodeLlama-13b-hf \
    --train_data_version v1.5_with_ir \
    --test_data_version v1.5_with_ir \
    --ckpt 500 \
    --device 3 \
    --use_lora_model True

# python performance_test.py \
#     --model_name CodeLlama-13b-hf \
#     --train_data_version None \
#     --test_data_version test \
#     --ckpt 500 \
#     --device 5 \
#     # --use_lora_model False