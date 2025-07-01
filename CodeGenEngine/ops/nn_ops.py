from tvm import (te,
                 auto_scheduler,
                 runtime,
                 topi)
import tvm

@auto_scheduler.register_workload
def adaptive_pool_avg(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    output_size = (8, 8)
    out = topi.nn.adaptive_pool(data, output_size, layout='NCHW', pool_type='avg')
    return [data,out]

@auto_scheduler.register_workload
def adaptive_pool_max(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    output_size = (8, 8)
    out = topi.nn.adaptive_pool(data, output_size, layout='NCHW', pool_type='max')
    return [data,out]

@auto_scheduler.register_workload
def fast_softmax(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    output = topi.nn.fast_softmax(data)
    return [data,output]

@auto_scheduler.register_workload
def add(N,C,H,W):
    data_a = te.placeholder((N,C,H,W), name='data')
    data_b = te.placeholder((N,C,H,W), name='data')
    out = tvm.topi.nn.add(data_a, data_b)
    return [data_a,data_b,out]

@auto_scheduler.register_workload
def batch_to_space_nd(N,C,H,W):
    data = te.placeholder((N, C, H, W), name='data')
    block_shape = [2, 2]  # 在空间维度上的块大小
    crop_begin = [0, 0]  # 开始的裁剪位置
    crop_end = [0, 0]  # 结束的裁剪位置

    out = tvm.topi.nn.batch_to_space_nd(data, block_shape, crop_begin, crop_end)
    return [data,out]

@auto_scheduler.register_workload
def global_pool_max(N,C,H,W):
    
    data = te.placeholder((N,C,H,W), name='data')
    pool_type = 'max'  
    output = tvm.topi.nn.global_pool(data, pool_type)
    return [data,output]

@auto_scheduler.register_workload
def global_pool_avg(N,C,H,W):
    
    data = te.placeholder((N,C,H,W), name='data')
    pool_type = 'avg' 
    output = tvm.topi.nn.global_pool(data, pool_type)
    return [data,output]

@auto_scheduler.register_workload
def dilate(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    output = topi.nn.dilate(data, (1,1,1,1),1)
    return [data,output]

@auto_scheduler.register_workload
def flatten(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    output = topi.nn.flatten(data)
    return [data,output]

@auto_scheduler.register_workload
def fifo_buffer(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    buffer = te.placeholder((N,C,H,W), name='buffer', dtype='float32')
    axis = 0  # 选择一个轴
    output = topi.nn.fifo_buffer(data,buffer,axis)
    return [data,buffer,output]

@auto_scheduler.register_workload
def conv2d_winograd_weight_transform(N, C,H, W):
    data = te.placeholder((N, C, W *2 , W *2), name="data")
    out = topi.nn.conv2d_winograd_weight_transform(data,2)
    return [data, out]

@auto_scheduler.register_workload
def concatenate(N,C,H,W):
    data_a = te.placeholder((N,C,H), name='data_a')
    data_b = te.placeholder((N,C,H), name='data_b')
    out = tvm.topi.nn.batch_matmul(data_a, data_b)

    conv = topi.nn.concatenate((data_a, data_b),0)
    return [data_a, data_b, conv]

@auto_scheduler.register_workload
def depth_to_space(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    block_size = 2
    layout = 'NCHW'
    mode = 'DCR'
    out = topi.nn.depth_to_space(data, block_size, layout, mode)
    return [data,out]

# 成功率低 设置较高的查询次数可能会有好的方案
# @auto_scheduler.register_workload
# def binarize_pack(N,C,H,W):
#     data = te.placeholder((N, C, H, W), name='data')
#     out = tvm.topi.nn.binarize_pack(data)
#     return [data,out]

#耗时太长了，暂时关闭
# @auto_scheduler.register_workload
# def binary_dense(N,C,H,W):
#     data = te.placeholder((N, W), name='data', dtype='uint32')
#     weight = te.placeholder((H, W), name='weight', dtype='uint32')
#     out = tvm.topi.nn.binary_dense(data, weight)
#     return [data,out]

@auto_scheduler.register_workload
def leaky_relu(N,C, H, W):
    data = te.placeholder((N, C, H, W), name="data")
    output = topi.nn.leaky_relu(data, 0.5)
    return [data,  output]
 
@auto_scheduler.register_workload
def log_softmax(N,C, H, W):
    data = te.placeholder((N, C, H, W), name="data")
    output = topi.nn.log_softmax(data)
    return [data,  output]

@auto_scheduler.register_workload
def lrn(N,C, H, W):
    data = te.placeholder((N, C, H, W), name="data")
    output = topi.nn.lrn(data,1)
    return [data,  output]

@auto_scheduler.register_workload
def mirror_pad(N,C, H, W):
    data = te.placeholder((H, W), name="data")
    pad_before = [1, 2]  # 在每个维度的起始位置进行填充的数量
    pad_after = [2, 1]   # 在每个维度的结束位置进行填充的数量
    out = tvm.topi.nn.mirror_pad(data, pad_before, pad_after)
    return [data,out]

@auto_scheduler.register_workload
def pad(N,C, H, W):
    data = te.placeholder((H, W), name="data")
    pad_before = [1, 2]  # 在每个维度的起始位置进行填充的数量
    pad_after = [2, 1]   # 在每个维度的结束位置进行填充的数量
    pad_value = 0.0       # 填充值为0.0

    out = tvm.topi.nn.pad(data, pad_before, pad_after, pad_value)
    return [data,out]

@auto_scheduler.register_workload
def pool1d(N,C, H, W):
    data = te.placeholder((N,C,W), name="data")
    # 定义池化参数
    kernel = 3  # 池化核的大小
    stride = 2  # 步幅
    dilation = 1  # 空洞卷积的空洞大小
    padding = 1  # 填充大小
    pool_type = "max"  # 池化类型，可以是"max"或"avg"
    ceil_mode = False  # 是否使用向上取整模式
    layout = "NCW"  # 输入数据布局
    count_include_pad = True  # 是否计算填充值
    out = tvm.topi.nn.pool1d(data, kernel, stride, (1,1), (1,1), pool_type, ceil_mode)
    return [data,out]

@auto_scheduler.register_workload
def pool2d(N,C, H, W):
    data = te.placeholder((N,C,H,W * 3), name="data")
    # 定义池化参数
    pool_type = "max"  # 池化类型，可以是"max"或"avg"
    ceil_mode = False  # 是否使用向上取整模式
    out = tvm.topi.nn.pool2d(data, (3,3), (2,2),  (1,1), (1,1,1,1),pool_type, ceil_mode)
    return [data,out]

@auto_scheduler.register_workload
def pool3d(N,C, H, W):
    data = te.placeholder((N,C,H,W ,16), name="data")
    # 定义池化参数
    kernel = (3, 3, 3)  # 池化核的大小
    stride = (2, 2, 2)  # 步幅
    dilation = (1, 1, 1)  # 空洞卷积的空洞大小
    padding = (1, 1, 1, 1, 1, 1)  # 填充大小
    pool_type = "max"  # 池化类型，可以是"max"或"avg"
    ceil_mode = False  # 是否使用向上取整模式
    layout = "NCDHW"  # 输入数据布局
    count_include_pad = True  # 是否计算填充值

    out = tvm.topi.nn.pool3d(data, kernel, stride, dilation, padding, pool_type, ceil_mode, layout, count_include_pad)
    return [data,out]

@auto_scheduler.register_workload
def relu(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.nn.relu(data)
    return [data,out]

@auto_scheduler.register_workload
def scale_shift_nchw(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    # 创建缩放因子张量
    scale_shape = (C,)  # 与通道数相同
    Scale = te.placeholder(scale_shape, name="Scale", dtype="float32")
    # 创建偏移量张量
    shift_shape = (C,)  
    Shift = te.placeholder(shift_shape, name="Shift", dtype="float32")
    out = topi.nn.scale_shift_nchw(data, Scale, Shift)
    return [data,Scale, Shift,out]

@auto_scheduler.register_workload
def prelu(N,C, H, W):

    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    # 创建缩放因子张量
    slope_shape = (W,)  
    slope = te.placeholder(slope_shape, name="Scale", dtype="float32")
    out = topi.nn.prelu(data, slope,axis=3)
    return [data,slope,out]

@auto_scheduler.register_workload
def scale_shift_nchwc(N,C, H, W):

    data = te.placeholder((N,2,H,W,C), name="data", dtype="float32")
    # 创建缩放因子张量
    scale_shape = (2,C,)  # 与通道数相同
    Scale = te.placeholder(scale_shape, name="Scale", dtype="float32")
    # 创建偏移量张量
    shift_shape = (2,C,)  
    Shift = te.placeholder(shift_shape, name="Shift", dtype="float32")
    out = topi.nn.scale_shift_nchwc(data, Scale, Shift)
    return [data,Scale, Shift,out]

@auto_scheduler.register_workload
def softmax(N,C, H, W):

    data = te.placeholder((N, C, H, W), name="data")
    output = topi.nn.softmax(data)
    return [data,  output]

@auto_scheduler.register_workload
def softmax_common(N,C, H, W):

    data = te.placeholder((N, C, H, W), name="data")
    output = topi.nn.softmax_common(data,axis=3,use_fast_exp=True)
    return [data,  output]

@auto_scheduler.register_workload
def space_to_depth(N,C, H, W):

    data_shape = (N, C, H * 2, W * 2)  # 输入张量的形状，包括批量大小、通道数、高度和宽度
    data = te.placeholder(data_shape, name="data", dtype="float32")
    block_size = 2  
    output = topi.nn.space_to_depth(data, block_size)
    return [data,  output]

@auto_scheduler.register_workload
def strided_slice(N,C, H, W):

    data_shape = (C,H,W)  # 输入张量的形状
    a = te.placeholder(data_shape, name="a", dtype="float32")

    # 定义切片参数
    begin = [1, 2, 3]  # 开始位置
    end = [4, 7, 10]   # 结束位置
    strides = None     # 步长，默认为None表示全为1
    axes = None        # 切片的轴，默认为None表示所有轴
    slice_mode = 'end' # 切片模式，默认为'end'

    # 调用tvm.topi.nn.strided_slice函数进行切片操作
    output = topi.nn.strided_slice(a, begin, end, strides, axes, slice_mode)
    return [a, output]

@auto_scheduler.register_workload
def unpack_NCHWc_to_nchw(N,C, H, W):
    packed_out_shape = (N, C, H, W, 2)  # 输入张量的形状，包括批量大小、通道数、高度、宽度和每个通道的元素数
    packed_out = te.placeholder(packed_out_shape, name="packed_out", dtype="float32")
    # 指定输出的数据类型
    out_dtype = "float32"  # 输出张量的数据类型

    # 将NCHWc格式的张量解包成普通的NCHW格式
    output = topi.nn.unpack_NCHWc_to_nchw(packed_out, out_dtype)

    return [packed_out, output]

@auto_scheduler.register_workload
def upsampling(N,C, H, W):
    data_shape = (N,C,H * 2 ,W * 2)  # 输入张量的形状，包括批量大小、通道数、高度和宽度
    data = te.placeholder(data_shape, name="data", dtype="float32")
    # 指定上采样比例
    scale_h = 2  # 垂直方向的上采样比例
    scale_w = 2  # 水平方向的上采样比例

    # 调用tvm.topi.nn.upsampling函数进行上采样操作
    output = topi.nn.upsampling(data, scale_h, scale_w)

    return [data, output]

@auto_scheduler.register_workload
def rms_norm(N,C, H, W):

    data_shape = (N,C,H, W)  # 输入张量的形状，包括批量大小、通道数、高度和宽度
    data = te.placeholder(data_shape, name="data", dtype="float32")
    weight_shape = (W,)  # 权重的形状，与通道数相同
    weight = te.placeholder(weight_shape, name="weight", dtype="float32")
    axis = (1,)
    epsilon = 1e-5
    # 调用 tvm.topi.nn.rms_norm 函数进行 rms 归一化操作
    output = topi.nn.rms_norm(data, weight, axis, epsilon)

    return [data, weight, output]

@auto_scheduler.register_workload
def bitserial_dense(N,C,H,W):
    kw = (W%10) * 32 
    kh = (H%10) * 8 
    kc = (C%10) * 8 
    data = te.placeholder((C, kw), name='data', dtype='uint32')
    weight = te.placeholder((kh, kw), name='weight', dtype='uint32')
    conv = topi.nn.bitserial_dense(data, weight ,8,8)
    return [data, weight, conv]

@auto_scheduler.register_workload
def batch_norm(N,C,H,W):
    data = te.placeholder((N, C, H, W), name='data')
    gamma = te.placeholder((C,), name='gamma')
    beta = te.placeholder((C,), name='beta')
    moving_mean = te.placeholder((C,), name='moving_mean')
    moving_var = te.placeholder((C,), name='moving_var')

    axis = 1  # 通常 batch_norm 的轴为通道轴，即 axis=1
    epsilon = 1e-5  # 设定一个很小的数作为 epsilon
    center = False  # 是否进行中心化
    scale = False  # 是否进行缩放
    training = False  # 是否在训练模式下
    momentum = 0.9  # 动量参数

    out = tvm.topi.nn.batch_norm(data, gamma, beta, moving_mean, moving_var, axis, epsilon, center, scale, training, momentum)
    return [data,gamma,beta,moving_mean,moving_var,out[0]]

@auto_scheduler.register_workload
def batch_matmul(batch, K,M,N):
    data_a = te.placeholder((batch, K,M), name='data_a')
    data_b = te.placeholder((batch, N,M), name='data_b')
    out = tvm.topi.nn.batch_matmul(data_a, data_b)
    return [data_a,data_b,out]

# @auto_scheduler.register_workload
# def matmul(batch, K,M,N):
#     data_a = te.placeholder((batch, K), name='data_a')
#     data_b = te.placeholder((K,M), name='data_b')
#     out = tvm.topi.nn.matmul(data_a, data_b)
#     return [data_a,data_b,out]

NN_OPS_LIST = [adaptive_pool_max,add,adaptive_pool_avg,fast_softmax,batch_to_space_nd,
               global_pool_max,global_pool_avg,dilate,flatten,fifo_buffer,
               conv2d_winograd_weight_transform,concatenate,depth_to_space,leaky_relu,log_softmax,
               lrn,mirror_pad,pad,pool1d,pool2d,
               pool3d,relu,scale_shift_nchw,prelu,scale_shift_nchwc,
               softmax,softmax_common,space_to_depth,strided_slice,unpack_NCHWc_to_nchw,
               upsampling,rms_norm,bitserial_dense,batch_norm]