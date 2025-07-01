set -e

export CUDA_VISIBLE_DEVICES=1

python main.py --model_name MODEL_NAME --level LEVEL --action ACTION --device cuda:0 --range left --use_lora True
