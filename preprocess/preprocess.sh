python preprocess.py \
    --use_ir True \
    --SAMPLING_RATIO 0.95 \
    --VERSION_INFO v2.0_nn_topi_alpaca \
    --use_alpaca True\
    --describe 根据算子名对数据集进行划分,同时包含算子的IR信息,包含topi,nn算子和alpaca数据集 \
    --alpaca_path ./raw_data/alpaca_data.json
