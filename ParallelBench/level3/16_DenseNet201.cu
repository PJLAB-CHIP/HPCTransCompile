#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void batch_norm_warp_kernel(
    float* output, const float* input,
    const float* weight, const float* bias,
    const float* mean, const float* var,
    int N, int C, int H, int W) {

    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int idx = global_tid; idx < N * C * H * W; idx += stride) {
        int c = (idx / (H * W)) % C;
        float inv_var = rsqrtf(var[c] + 1e-5f);

        // Load data into shared memory
        shared_mem[tid] = input[idx];
        __syncthreads();

        // Warp-level reduction for local sum
        float local_sum = shared_mem[tid];
        local_sum = warp_reduce_sum(local_sum);

        // First thread in warp writes result
        if (tid % WARP_SIZE == 0) {
            shared_mem[tid / WARP_SIZE] = local_sum;
        }
        __syncthreads();

        if (tid < WARP_SIZE) {
            local_sum = (tid < blockDim.x / WARP_SIZE) ? shared_mem[tid] : 0.0f;
            local_sum = warp_reduce_sum(local_sum);

            if (tid == 0) {
                shared_mem[0] = local_sum;
            }
        }
        __syncthreads();

        // Normalize using the computed statistics
        float normalized = (input[idx] - mean[c]) * inv_var;
        output[idx] = weight[c] * normalized + bias[c];
    }
}

torch::Tensor dense_layer_fn(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor conv_weight,
    bool is_training) {

    auto sizes = x.sizes();
    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];

    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    auto output = torch::empty_like(x);

    if (!is_training) {
        batch_norm_warp_kernel<<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(),
            bn_var.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        output = at::batch_norm(x, bn_weight, bn_bias, bn_mean, bn_var, is_training, 0.1, 1e-5, true);
    }

    output = at::relu(output);
    output = at::conv2d(output,
                       conv_weight,
                       c10::nullopt,
                       at::IntArrayRef(std::vector<int64_t>{1, 1}),
                       at::IntArrayRef(std::vector<int64_t>{1, 1}));
    output = at::dropout(output, 0.0, is_training);
    return output;
}

torch::Tensor dense_block_fn(torch::Tensor x, pybind11::list layer_params, bool is_training) {
    std::vector<torch::Tensor> features;
    features.push_back(x);

    for (ssize_t i = 0; i < layer_params.size(); i++) {
        auto params_tuple = layer_params[i].cast<pybind11::tuple>();
        torch::Tensor bn_weight = params_tuple[0].cast<torch::Tensor>();
        torch::Tensor bn_bias = params_tuple[1].cast<torch::Tensor>();
        torch::Tensor bn_mean = params_tuple[2].cast<torch::Tensor>();
        torch::Tensor bn_var = params_tuple[3].cast<torch::Tensor>();
        torch::Tensor conv_weight = params_tuple[4].cast<torch::Tensor>();

        torch::Tensor new_feature = dense_layer_fn(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, is_training);
        features.push_back(new_feature);
        x = at::cat(features, 1);
    }
    return x;
}

torch::Tensor transition_layer_fn(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor conv_weight,
    bool is_training) {

    auto sizes = x.sizes();
    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];

    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    auto output = torch::empty_like(x);

    if (!is_training) {
        batch_norm_warp_kernel<<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(),
            bn_var.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        output = at::batch_norm(x, bn_weight, bn_bias, bn_mean, bn_var, is_training, 0.1, 1e-5, true);
    }

    output = at::relu(output);
    output = at::conv2d(output,
                     conv_weight,
                     c10::nullopt,
                     at::IntArrayRef(std::vector<int64_t>{1, 1}),
                     at::IntArrayRef(std::vector<int64_t>{0, 0}));
    output = at::avg_pool2d(output,
                         at::IntArrayRef(std::vector<int64_t>{2, 2}),
                         at::IntArrayRef(std::vector<int64_t>{2, 2}));
    return output;
}

torch::Tensor forward(torch::Tensor x, pybind11::object params_obj, bool is_training) {
    pybind11::dict params = params_obj.cast<pybind11::dict>();

    torch::Tensor features_conv_weight = params["features_conv_weight"].cast<torch::Tensor>();
    torch::Tensor features_bn_mean = params["features_bn_mean"].cast<torch::Tensor>();
    torch::Tensor features_bn_var = params["features_bn_var"].cast<torch::Tensor>();
    torch::Tensor features_bn_weight = params["features_bn_weight"].cast<torch::Tensor>();
    torch::Tensor features_bn_bias = params["features_bn_bias"].cast<torch::Tensor>();

    x = at::conv2d(x,
                 features_conv_weight,
                 c10::nullopt,
                 at::IntArrayRef(std::vector<int64_t>{2, 2}),
                 at::IntArrayRef(std::vector<int64_t>{3, 3}));

    auto sizes = x.sizes();
    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);

    auto output = torch::empty_like(x);
    if (!is_training) {
        batch_norm_warp_kernel<<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            features_bn_weight.data_ptr<float>(),
            features_bn_bias.data_ptr<float>(),
            features_bn_mean.data_ptr<float>(),
            features_bn_var.data_ptr<float>(),
            N, C, H, W
        );
        x = output;
    } else {
        x = at::batch_norm(x, features_bn_weight, features_bn_bias, 
                          features_bn_mean, features_bn_var, 
                          is_training, 0.1, 1e-5, true);
    }

    x = at::relu(x);
    x = at::max_pool2d(x,
                     at::IntArrayRef(std::vector<int64_t>{3, 3}),
                     at::IntArrayRef(std::vector<int64_t>{2, 2}),
                     at::IntArrayRef(std::vector<int64_t>{1, 1}));

    pybind11::list dense_blocks = params["dense_blocks"].cast<pybind11::list>();
    pybind11::list transition_layers = params["transition_layers"].cast<pybind11::list>();

    int num_dense_blocks = dense_blocks.size();
    for (int i = 0; i < num_dense_blocks; i++) {
        pybind11::list block_params = dense_blocks[i].cast<pybind11::list>();
        x = dense_block_fn(x, block_params, is_training);

        if (i != num_dense_blocks - 1) {
            auto trans_tuple = transition_layers[i].cast<pybind11::tuple>();
            torch::Tensor t_bn_weight = trans_tuple[0].cast<torch::Tensor>();
            torch::Tensor t_bn_bias = trans_tuple[1].cast<torch::Tensor>();
            torch::Tensor t_bn_mean = trans_tuple[2].cast<torch::Tensor>();
            torch::Tensor t_bn_var = trans_tuple[3].cast<torch::Tensor>();
            torch::Tensor t_conv_weight = trans_tuple[4].cast<torch::Tensor>();

            x = transition_layer_fn(x, t_bn_weight, t_bn_bias, t_bn_mean, 
                                  t_bn_var, t_conv_weight, is_training);
        }
    }

    torch::Tensor final_bn_mean = params["final_bn_mean"].cast<torch::Tensor>();
    torch::Tensor final_bn_var = params["final_bn_var"].cast<torch::Tensor>();
    torch::Tensor final_bn_weight = params["final_bn_weight"].cast<torch::Tensor>();
    torch::Tensor final_bn_bias = params["final_bn_bias"].cast<torch::Tensor>();

    sizes = x.sizes();
    N = sizes[0]; C = sizes[1]; H = sizes[2]; W = sizes[3];
    output = torch::empty_like(x);

    if (!is_training) {
        batch_norm_warp_kernel<<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            final_bn_weight.data_ptr<float>(),
            final_bn_bias.data_ptr<float>(),
            final_bn_mean.data_ptr<float>(),
            final_bn_var.data_ptr<float>(),
            N, C, H, W
        );
        x = output;
    } else {
        x = at::batch_norm(x, final_bn_weight, final_bn_bias,
                          final_bn_mean, final_bn_var,
                          is_training, 0.1, 1e-5, true);
    }

    x = at::relu(x);
    x = at::adaptive_avg_pool2d(x, at::IntArrayRef(std::vector<int64_t>{1, 1}));
    x = x.view({x.size(0), -1});

    torch::Tensor classifier_weight = params["classifier_weight"].cast<torch::Tensor>();
    torch::Tensor classifier_bias = params["classifier_bias"].cast<torch::Tensor>();
    x = at::linear(x, classifier_weight, classifier_bias);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CUDA forward function");
}
