import torch
from torch.utils.cpp_extension import load_inline, load

# Python 不允许模块名以数字开头。如果文件名为 1_Square_matrix_multiplication_.py，
# 需重命名为合法的模块名，比如 Square_matrix_multiplication_.py
# from Square_matrix_multiplication_ import Model,get_inputs,get_init_inputs
# from conv_transposed_3D__asymmetric_input__square_kernel import (
#     Model,
#     get_inputs,
#     get_init_inputs,
# )
from _38_LTSMBidirectional import (
    Model,
    get_inputs,
    get_init_inputs,
)

# 将cuda_path修改为自己编写的.cu
cuda_path = "OpSample/38_LTSMBidirectional.cu"

TEST_TIMES = 1


def eval(eval_module, torch_model):
    for i in range(TEST_TIMES):
        inputs = get_inputs()
        inputs = [tensor.to("cuda") for tensor in inputs]
        torch_result = torch_model(*inputs)
        cuda_result = eval_module.forward(*inputs)
        if torch.allclose(torch_result, cuda_result, rtol=1e-02, atol=1e-02):
            continue
        else:
            print("Not Pass!")
    print("Pass!")


if __name__ == "__main__":
    eval_module = load(name="test", sources=[cuda_path], verbose=True)
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).cuda()
    eval(eval_module, model)
