# Dataset Overview

Generated Graph Num: 70
Model Num: 7
Single Op Num: 77(+3)

注：单算子中少了rsqrt、sinh和cosh，这三个算子在提供的single_v1_0数据集中，提取其中一条加入到benchmark中

# Single Op Information

ADDITIONAL_OP_TYPE_DICT = {
    'Elementwise':[],
    'Reduction':[],
    'Layout Transform':['reshape', 'transpose', 'gather_nd', 'scatter_nd', 'reorg'],
    'Logic Intensive':['resize2d', 'resize3d', 'grid_sample', 'argsort'],
    'Compute Intensive':['conv2d_opt', 'dft', 'group_conv2d_opt', 'batch_matmul_opt']
}

OP_TYPE_DICT = {
    'Elementwise':['erf', 'leaky_relu', 'sqrt', 'asin', 'tanh', 'floor', 'log', 'sin', 'prelu', 'fast_exp', 'log2', 'sign', 'sigmoid', 'isnan', 'relu', 'cos', 'fast_tanh', 'log10', 'round', 'atan', 'negative', 'add', 'tan', 'atanh', 'acos', 'ceil', 'asinh', 'abs', 'exp','rsqrt', 'sinh', 'cosh', 'combination_op', 'fast_erf'],
    'Reduction':['min', 'global_pool_max', 'global_pool_avg', 'sum', 'max', 'rms_norm', 'log_softmax', 'softmax', 'fast_softmax', 'softmax_common', 'prod', 'pool3d', 'pool1d', 'pool2d',  'adaptive_pool_max', 'adaptive_pool_avg'],
    'Layout Transform':['unpack_NCHWc_to_nchw', 'flatten', 'scale_shift_nchw', 'flip', 'depth_to_space', 'batch_to_space_nd', 'strided_slice', 'space_to_depth', 'scale_shift_nchwc', 'mirror_pad', 'dilate'],
    'Logic Intensive':['fifo_buffer', 'multi_out_op', 'shape', 'upsampling'],
    'Compute Intensive':['lrn', 'matmul']
}