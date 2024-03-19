###############################################
# train_data_version 和 ckpt 指定了要使用的model
# test_data_version 指定了测试的数据集
###############################################

# LORA √ ALPACA_PROMPT √
# 更改model_name+train_data_version+ckpt更换模型
# 更改test_data_version更换测试集

####1####
# python performance_test.py \
#     --model_name CodeLlama-13b-hf \
#     --train_data_version v3.0_nn_topi_without_ir \
#     --test_data_version v3.0_nn_topi_without_ir \
#     --ckpt 50 \
#     --device 0 \
#     --max_length 3000 \
#     --use_lora_model True \
#     --bits 8 \
    
####2####
python performance_test.py \
    --model_name CodeLlama-13b-hf \
    --train_data_version v3.0_topi_without_ir_by_name_simplify \
    --test_data_version v3.0_topi_without_ir_by_name_simplify \
    --ckpt 400 \
    --device 2 \
    --max_length 4096 \
    --extra_length 500 \
    --use_lora_model True \
    --quantize_model False \
    --bits 8 \
    --prompt_format none \
    --match_output True \
