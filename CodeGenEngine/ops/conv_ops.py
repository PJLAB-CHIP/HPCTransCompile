from tvm import (te,
                 auto_scheduler,
                 runtime,
                 topi)
import tvm

@auto_scheduler.register_workload
def conv2d_nchw(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, bias, conv]


@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N,CI,H,W), name='data', dtype='float32')
    kernel = te.placeholder((CO,CI,KH,KW), name='kernel', dtype='float32')
    dilation = (1, 1)  
    out = tvm.topi.nn.conv2d(data, kernel, stride, padding, dilation, data_layout='NCHW', out_dtype='float32')
    return [data, kernel,  out]


@auto_scheduler.register_workload
def conv1d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((2,3,10), name='data', dtype='float32')
    kernel = te.placeholder((5,3,3), name='kernel', dtype='float32')
    out = topi.nn.conv1d(data, kernel, strides=2, padding='VALID', dilation=1, data_layout='NCW')
    # data = te.placeholder((N,CI,W), name='data', dtype='float32')
    # kernel = te.placeholder((CO,CI,KW), name='kernel', dtype='float32')
    # out = topi.nn.conv1d(data, kernel, strides=2, padding='VALID', dilation=1, data_layout='NCW')
    return [data, kernel, out]

@auto_scheduler.register_workload
def conv1d_ncw(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N,CI,W), name='data', dtype='float32')
    kernel = te.placeholder((CO,KW,KW), name='kernel', dtype='float32')
    out = topi.nn.conv1d_ncw(data, kernel, strides=2, padding='VALID', dilation=1)
    return [data, kernel,  out]

@auto_scheduler.register_workload
def conv1d_transpose_ncw(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N,W,CI), name='data', dtype='float32')
    kernel = te.placeholder((CO,CI,KW), name='kernel', dtype='float32')
    out = topi.nn.conv1d_transpose_ncw(data, kernel, stride=2, padding='VALID',out_dtype = 'float32',output_padding=0)
    return [data, kernel,  out]

@auto_scheduler.register_workload
def conv2d_gemm_weight_transform(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name='data', dtype='float32')
    kernel = te.placeholder((CO, CI, KH, KW), name='kernel', dtype='float32')
    tile_N = 4
    tile_K = 16
    out = tvm.topi.nn.conv2d_gemm_weight_transform(kernel, tile_N, tile_K)

@auto_scheduler.register_workload
def conv3d_ncdhw(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI,CO, H, W), name="data")
    kernel = te.placeholder((CO, CI,N, KH, KW), name="kernel")
    conv = topi.nn.conv3d_ncdhw(data, kernel, 1, 1, dilation=1, out_dtype="float32",groups=1)
    return [data, kernel, conv]

# @auto_scheduler.register_workload
# def conv3d_ndhwc(N, H, W, CO, CI, KH, KW, stride, padding):
#     data = te.placeholder((N, CO, H, W,CI), name="data")
#     kernel = te.placeholder((CO,N, KH, KW, CI), name="kernel")
#     conv = topi.nn.conv3d_ndhwc(data, kernel, 1, 1, dilation=1, out_dtype="float32",groups=1)
#     return [data, kernel, conv]

@auto_scheduler.register_workload
def conv3d_winograd_weight_transform(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N,CI, CO, W*2, W*2), name="data")
    conv = topi.nn.conv3d_winograd_weight_transform(data, 2)
    return [data, conv]

@auto_scheduler.register_workload
def depthwise_conv2d_nchw(N, H, W, CO, CI, KH, KW, stride, padding):
    
    data = te.placeholder((N,CI,CO,H), name='data')
    kernel = te.placeholder((CI,CO,KH,KW), name='kernel', dtype='float32')
    output = topi.nn.depthwise_conv2d_nchw(data, kernel,1,padding='VALID',dilation=1)
    return [data,kernel,output]

@auto_scheduler.register_workload
def depthwise_conv2d_nhwc(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N,H,CO,CI), name='data')
    kernel = te.placeholder((KH,KW,CO), name='kernel', dtype='float32')
    output = topi.nn.depthwise_conv2d_nhwc(data, kernel,1,padding='VALID',dilation=1)
    return [data,kernel,output]

@auto_scheduler.register_workload
def group_conv1d_ncw(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI,  W), name="data")
    kernel = te.placeholder((CO, CI,  KW), name="kernel")
    strides = 1
    padding = 'VALID'
    dilation = 1
    groups = 1
    out_dtype = 'float32'  # 输出数据类型，如果不设置则为输入数据的类型
    output = tvm.topi.nn.group_conv1d_ncw(data, kernel, strides, padding, dilation, groups, out_dtype)
    return [data,kernel,output]

@auto_scheduler.register_workload
def group_conv1d_nwc(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N,  W, CI), name="data")
    kernel = te.placeholder((CO, CI,  KW), name="kernel")
    strides = 1
    padding = 'VALID'
    dilation = 1
    groups = 1
    out_dtype = 'float32'  # 输出数据类型，如果不设置则为输入数据的类型
    output = tvm.topi.nn.group_conv1d_nwc(data, kernel, strides, padding, dilation, groups, out_dtype)
    return [data,kernel,output]


# @auto_scheduler.register_workload
# def group_conv2d_nchw(N, H, W, CO, CI, KH, KW, stride, padding):
    
#     data = te.placeholder((N, CI, H, W), name="data")
#     kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
#     bias = te.placeholder((1, CO, 1, 1), name="bias")
#     conv = topi.nn.group_conv2d_nchw(data, kernel, stride, padding, dilation=1,groups=1, out_dtype="float32")
#     return [data, kernel, bias, conv]

CONV_OPS_LIST = [conv1d,conv2d_nchw,conv2d,conv1d_ncw,conv1d_transpose_ncw,
                 conv2d_gemm_weight_transform,conv3d_ncdhw,conv3d_winograd_weight_transform,depthwise_conv2d_nchw,depthwise_conv2d_nhwc,
                 group_conv1d_ncw,group_conv1d_nwc]