#### QwenCoder_14b ####

set -e

export CUDA_VISIBLE_DEVICES=1

# # tmux opencoder-infer
# python main.py --model_name OpenCoder --level level1 --action translation_c --device cuda:1 --range left
# python main.py --model_name OpenCoder --level level2 --action translation_c --device cuda:1 --range left  --use_lora True

# # tmux opencoder-infer-2 [DONE]
# python main.py --model_name OpenCoder --level level3 --action translation_c --device cuda:2 --range left --use_lora True

# # tmux opencoder-infer-3
python main.py --model_name DeepSeekCoder_Lite --level level1 --action translation_c --device cuda:0 --range left --use_lora True
# python main.py --model_name DeepSeekCoder_Lite --level level2 --action translation_c --device cuda:4 --range left --use_lora True

# # tmux opencoder-infer-4
# python main.py --model_name DeepSeekCoder_Lite --level level3 --action translation_c --device cuda:5 --range left --use_lora True
# python main.py --model_name DeepSeekCoder_Lite --level level1 --action translation_c --device cuda:5 --range left
# python main.py --model_name DeepSeekCoder_Lite --level level2 --action translation_c --device cuda:5 --range left
# python main.py --model_name DeepSeekCoder_Lite --level level3 --action translation_c --device cuda:5 --range left

# python main.py --model_name QwenCoder_14b --level level2 --action translation_c --device cuda:4 --range all
# python main.py --model_name QwenCoder_14b --level level3 --action translation_c --device cuda:5 --range all

# python main.py --model_name QwenCoder_14b --level level1 --action translation_c --device cuda:3 --range all --use_lora True
# python main.py --model_name QwenCoder_14b --level level2 --action translation_c --device cuda:4 --range all --use_lora True
# python main.py --model_name QwenCoder_14b --level level3 --action translation_c --device cuda:5 --range all --use_lora True

# nohup python main.py --model_name QwenCoder_14b --level level1 --action optimization_c --device cuda:0 > log/level1.log 2>&1 &
# nohup python main.py --model_name QwenCoder_14b --level level2 --action optimization_c --device cuda:1 > log/level2.log 2>&1 &
# nohup python main.py --model_name QwenCoder_14b --level level3 --action optimization_c --device cuda:2 > log/level3.log 2>&1 &

# wait

#### DeepSeekCoder_Lite ####
#!/bin/bash

# nohup python main.py --model_name DeepSeekCoder_Lite --level level1 --action translation_c --device cuda:6 > log/DeepSeekCoder_Lite_level1_round2.log 2>&1 &
# nohup python main.py --model_name DeepSeekCoder_Lite --level level2 --action translation_c --device cuda:4 > log/DeepSeekCoder_Lite_level2.log 2>&1 &
# nohup python main.py --model_name DeepSeekCoder_Lite --level level3 --action translation_c --device cuda:5 > log/DeepSeekCoder_Lite_level3.log 2>&1 &

# wait

#### LingCoder ####
# python main.py --model_name LingCoder --level level1 --action translation_c --device cuda:5 --range left


#### OpenCoder ####
# python main.py --model_name OpenCoder --level level1 --action translation_c --device cuda:4 --range all
