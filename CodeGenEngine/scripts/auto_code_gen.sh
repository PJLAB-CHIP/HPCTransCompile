#!/bin/bash
PYTHON_PROGRAM="code_gen_single.py"

MAX_RETRIES=5

retries=0

TIMEOUT=3600

NN_OPS_LIST=(adaptive_pool_max add adaptive_pool_avg fast_softmax batch_to_space_nd 
               global_pool_max global_pool_avg dilate flatten fifo_buffer 
               conv2d_winograd_weight_transform concatenate depth_to_space leaky_relu log_softmax 
               lrn mirror_pad pad pool1d pool2d 
               pool3d relu scale_shift_nchw prelu scale_shift_nchwc 
               softmax softmax_common space_to_depth strided_slice unpack_NCHWc_to_nchw 
               upsampling rms_norm bitserial_dense batch_norm)

TOPI_OPS_LIST=(abs cos atan clip asin_cos sum cosh acos asin asinh atanh ceil 
                 const_vector const_vector const_vector const_vector erf exp fast_erf fast_exp fast_tanh fixed_point_multiply 
                 flip floor full_like isnan log log10 log2 max min negative 
                 prod reinterpret repeat reshape reshape round rsqrt shape sigmoid sign 
                 sin sinh sqrt tan tanh tile matmul combination_op multi_out_op)

CONV_OPS_LIST=(conv1d conv2d_nchw conv2d conv1d_ncw conv1d_transpose_ncw 
                 conv2d_gemm_weight_transform conv3d_ncdhw conv3d_winograd_weight_transform depthwise_conv2d_nchw depthwise_conv2d_nhwc 
                 group_conv1d_ncw group_conv1d_nwc)

#### topi ####
for op in "${TOPI_OPS_LIST[@]}"; do
    retries=0
    while true; do
        python $PYTHON_PROGRAM \
            --op_type topi \
            --timeout $TIMEOUT \
            --op_name $op \
#### nn ####
# for op in "${NN_OPS_LIST[@]}"; do
#     retries=0
#     while true; do
#         python $PYTHON_PROGRAM \
#             --op_type nn \
#             --timeout $TIMEOUT \
#             --op_name $op \
#### conv ####
# for op in "${CONV_OPS_LIST[@]}"; do
#     retries=0
#     while true; do
#         python $PYTHON_PROGRAM \
#             --op_type conv \
#             --timeout $TIMEOUT \
#             --op_name $op \
#### execution body ####
        # Check whether the program was successfully executed
        if [ $? -eq 0 ]; then
            break
        else
            retries=$((retries+1))
            if [ $retries -ge $MAX_RETRIES ]; then
                echo "Maximum number of retries reached. Exiting."
                break
            fi
            echo "Program failed. Retrying in 5 seconds..."
            sleep 5
        fi
    done
done
echo "All operations completed."