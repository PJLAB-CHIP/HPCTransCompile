python performance_test.py \
    --model_name CodeLlama-34b-Instruct-hf \
    --train_data_version None \
    --test_data_version v1.5_without_ir \
    --ckpt 500 \
    --device 0 \
    # --use_lora_model 0


# python performance_test.py \
#     --model_name CodeLlama-34b-Instruct-hf \
#     --train_data_version v1.5_without_ir \
#     --test_data_version v1.5_with_ir \
#     --ckpt 500 \
#     --device 3 \
#     --use_lora_model True