CUDA_VISIBLE_DEVICES=5 \
python train.py --config DeepCoder.yaml > log/log_deepcoder_14b.log
# python train.py --config DeepSeekCoder_Lite.yaml > log/log_deepseekcoder_lite.log
# python train.py > log/models--Qwen--Qwen2.5-Coder-14B-Instruct.log
# nohup python train.py > log/models--Qwen--Qwen2.5-Coder-14B-Instruct.log &
# nohup python train.py > log/DeepSeek-Coder-V2-Lite-Instruct.log &

