CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
python train.py --config OpenCoder.yaml > log/log_opencoder.log
# python train.py > log/models--Qwen--Qwen2.5-Coder-14B-Instruct.log
# nohup python train.py > log/models--Qwen--Qwen2.5-Coder-14B-Instruct.log &
# nohup python train.py > log/DeepSeek-Coder-V2-Lite-Instruct.log &

