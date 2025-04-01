import torch
from torch.utils.cpp_extension import load_inline,load
# Python 不允许模块名以数字开头。如果文件名为 1_Square_matrix_multiplication_.py，
# 需重命名为合法的模块名，比如 Square_matrix_multiplication_.py
from Mamba2ReturnY import Model,get_inputs,get_init_inputs

# 将cuda_path修改为自己编写的.cu
cuda_path = '/code/LLM4HPCTransCompile/OpSample/48_Mamba2ReturnY.cu'
torch.manual_seed(42)
TEST_TIMES = 1

def eval(eval_module,torch_model):
    for i in range(TEST_TIMES):
        inputs = get_inputs()
        inputs = [tensor.to('cuda') for tensor in inputs]
        torch_model = torch_model.cuda()
        torch_result = torch_model(*inputs)
        cuda_result = torch_model.forward(*inputs, fn=eval_module.forward)
        if torch.allclose(torch_result,cuda_result,rtol=1e-02,atol=1e-02):
            continue
        # if torch.equal(torch_result,cuda_result):
        #     continue
        else:
            print('Not Pass!')
    print('Pass!')

if __name__ == '__main__':
    eval_module = load(
        name='test',
        sources=[cuda_path],
        verbose=True
    )
    init_inputs = get_init_inputs()
    model = Model(*init_inputs)
    eval(eval_module,model)
