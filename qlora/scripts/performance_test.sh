###############################################
# train_data_version 和 ckpt 指定了要使用的model
# test_data_version 指定了测试的数据集
###############################################

# LORA √ ALPACA_PROMPT √
# 更改model_name+train_data_version+ckpt更换模型
# 更改test_data_version更换测试集

####1####
python performance_test.py \
    --model_name CodeLlama-13b-hf \
    --train_data_version v2.0_without_ir \
    --test_data_version test \
    --ckpt 500 \
    --device 0 \
    --max_length 50 \
    --use_lora_model True \
    --use_alpaca_prompt True \
    --bits 8 \
    
####2####
# python performance_test.py \
#     --model_name CodeLlama-13b-hf \
#     --train_data_version v1.5_with_ir \
#     --test_data_version v1.5_with_ir \
#     --ckpt 500 \
#     --device 1 \
#     --max_length 500 \
#     --use_lora_model True \
#     --use_alpaca_prompt True \
#     --bits 8 \