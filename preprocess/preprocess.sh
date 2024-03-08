##################################################
# Additional Modification Item
# utils.py
# raw_data_path: 原始算子数据路径
##################################################

python preprocess.py \
    --SAMPLING_RATIO 0.95 \
    --VERSION_INFO v2.1_simplify_nn_topi_without_ir \
    --describe 根据算子名对数据集进行划分,同时不包含算子的IR信息,包含topi,nn算子,不包含alpaca数据集 \
    --alpaca_path ./raw_data/alpaca_data.json \
    --extract_func_implementation True \
    # --use_ir False \
    # --use_alpaca False \

