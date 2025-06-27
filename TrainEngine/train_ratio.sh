CUDA_VISIBLE_DEVICES=0 \
# python train.py --config QwenCoder_14b.yaml --ratio 75 > log/ratio/log_QwenCoder_14b_75.log
python train.py --config DeepSeekCoder_Lite.yaml --ratio 25 > log/ratio/log_DeepSeekCoder_Lite_25.log


# python train.py --config DeepSeekCoder_Lite.yaml > log/log_deepseekcoder_lite.log
# python train.py > log/models--Qwen--Qwen2.5-Coder-14B-Instruct.log
# nohup python train.py > log/models--Qwen--Qwen2.5-Coder-14B-Instruct.log &
# nohup python train.py > log/DeepSeek-Coder-V2-Lite-Instruct.log &

