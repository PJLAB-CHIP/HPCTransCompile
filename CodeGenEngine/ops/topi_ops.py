"""element wise ops with single input(Tensor)"""
from tvm import (te,
                 auto_scheduler,
                 runtime,
                 topi)
import random
# element_wise_ops = ['abs']
# @auto_scheduler.register_workload
# def elemet_wise_op(N,C,H,W):
#     data = te.placeholder((N,C,H,W), name="data", dtype="float32")
#     exec(f'out = topi.{element_wise_ops[0]}(data)')
#     return [data,out]

# @auto_scheduler.register_workload
# def asin_cos(N,C,H,W):
#     data = te.placeholder((N,C,H,W), name="data", dtype="float32")
#     # out = topi.sum(data)
#     out = topi.asin(data)
#     out = topi.cos(out)
#     return [data,out]

@auto_scheduler.register_workload
def abs(N,C,H,W):
    data = te.placeholder((N,C,H,W),name='data',dtype='float32')
    out = topi.abs(data)
    return [data,out]

@auto_scheduler.register_workload
def sum(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sum(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def cosh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.cosh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def cos(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.cos(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def acos(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.acos(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def asin(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.asin(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def asinh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.asinh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def atan(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.atan(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def atanh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.atanh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def ceil(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.ceil(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def clip(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.clip(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def const_vector(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.const_vector(data)
    return [data,out]
    

@auto_scheduler.register_workload
def erf(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.erf(data)
    return [data,out]
    

@auto_scheduler.register_workload
def exp(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.exp(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fast_erf(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fast_erf(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fast_exp(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fast_exp(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fast_tanh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fast_tanh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fixed_point_multiply(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fixed_point_multiply(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def flip(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.flip(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def floor(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.floor(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def full_like(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.full_like(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def isnan(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.isnan(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def log(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.log(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def log10(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.log10(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def log2(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.log2(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def max(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.max(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def min(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.min(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def negative(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.negative(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def prod(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.prod(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def reinterpret(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.reinterpret(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def repeat(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.repeat(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def reshape(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.reshape(data)
    return [data,out]
    

    
@auto_scheduler.register_workload
def round(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.round(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def rsqrt(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.rsqrt(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def shape(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.shape(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sigmoid(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sigmoid(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sign(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sign(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sin(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sin(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sinh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sinh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sqrt(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sqrt(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def tan(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.tan(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def tanh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.tanh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def tile(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.tile(data)
    return [data,out]


@auto_scheduler.register_workload
def matmul(N,C,H,W):
    data_a = te.placeholder((H,W),name="left_matrix",dtype='float32')
    data_b = te.placeholder((W,H),name='right_matrix',dtype='float32')
    out = topi.matmul(data_a,data_b)
    return [data_a,data_b,out]

"""test combination ops"""

@auto_scheduler.register_workload
def combination_op(N,C,H,W):
    data_a = te.placeholder((N,C,H,W), name='data')
    data_b = te.placeholder((N,C,H,W), name='data')
    _data_a = topi.sqrt(data_a)
    _data_b = topi.cos(data_b)
    out = topi.nn.add(_data_a,_data_b)
    return [data_a,data_b,out]

@auto_scheduler.register_workload
def multi_out_op(N,C,H,W):
    data_a = te.placeholder((N,C,H,W), name='data')
    data_b = te.placeholder((N,C,H,W), name='data')
    out = topi.nn.add(data_a,data_b)
    out_1 = topi.sqrt(out)
    out_2 = topi.cos(out)
    return [data_a,data_b,out_1,out_2]

@auto_scheduler.register_workload
def abs_simple(data):
    out = topi.sum(data)
    print(out)
    return [data,out]


TOPI_OPS_LIST = [matmul,multi_out_op,combination_op,abs_simple,
                 abs,cos,atan,clip,sum,cosh,acos,asin,asinh,atanh,ceil,
                 const_vector,const_vector,const_vector,const_vector,erf,exp,fast_erf,fast_exp,fast_tanh,fixed_point_multiply,
                 flip,floor,full_like,isnan,log,log10,log2,max,min,negative,
                 prod,reinterpret,repeat,reshape,reshape,round,rsqrt,shape,sigmoid,sign,
                 sin,sinh,sqrt,tan,tanh,tile]

def generate_new_function():
    # 随机选择函数个数
    # num_functions = random.randint(1, len(TOPI_OPS_LIST))
    num_functions = 2
    
    # 随机选择函数
    selected_functions = random.sample(TOPI_OPS_LIST, num_functions)

    print(selected_functions)
    
    # 定义新函数
    def new_function(N,C,H,W):
        for idx,func in enumerate(selected_functions):
            if idx == 0:
                data,out = func(N,C,H,W)
                print(data,out)
            else:
                data,out = func()
        return [data,out]
    
    return new_function

if __name__ == '__main__':
    """test simple func"""
    data = te.placeholder((12,23,34,45),name='data',dtype='float32')
    func = abs_simple(data)
    print(func)
    """test combination ops"""
    # new_func = generate_new_function()
    # print(new_func(12,13,13,3))  # 输出新函数的结果