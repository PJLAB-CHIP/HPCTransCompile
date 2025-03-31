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
from _29_SwinMLP import (
    Model,
    get_inputs,
    get_init_inputs,
)

# 将cuda_path修改为自己编写的.cu
cuda_path = "OpSample/29_SwinMLP.cu"

TEST_TIMES = 100


def eval(eval_module, torch_model):
    for i in range(TEST_TIMES):
        inputs = get_inputs()
        # inputs = [tensor.to("cuda") for tensor in inputs]
        torch_result = torch_model(*inputs)
        """
        torch::Tensor norm_weight, torch::Tensor norm_bias,
        torch::Tensor head_weight, torch::Tensor head_bias, int64_t num_layers,
        int64_t embed_dim, std::vector<int64_t> depths,
        std::vector<int64_t> num_heads, int64_t window_size, float mlp_ratio,
        float drop_rate, std::pair<int64_t, int64_t> patches_resolution) {
        """
        cuda_result = eval_module.forward(
            *inputs,
            torch_model.patch_embed_params,
            torch_model.drop_rate,
            torch_model.layers_params,
            torch_model.norm_weight,
            torch_model.norm_bias,
            torch_model.head_weight,
            torch_model.head_bias,
            torch_model.img_size,
            torch_model.patch_size,
            torch_model.embed_dim,
            torch_model.depths,
            torch_model.num_heads,
            torch_model.window_size,
            torch_model.mlp_ratio,
            torch_model.drop_rate,
            torch_model.drop_path_rate,
            torch_model.use_checkpoint,
        )
        if torch.allclose(torch_result, cuda_result, rtol=1e-02, atol=1e-02):
            continue
        # if torch.equal(torch_result,cuda_result):
        #     continue
        else:
            print("Not Pass!")
    print("Pass!")


if __name__ == "__main__":
    eval_module = load(name="test", sources=[cuda_path], verbose=True)
    init_inputs = get_init_inputs()
    model = Model(*init_inputs)
    eval(eval_module, model)
