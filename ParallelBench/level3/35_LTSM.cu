#include <torch/extension.h>
#include <cmath>

namespace {

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void lstm_elementwise_stride(
    const float* __restrict__ gates,
    const float* __restrict__ prev_c,
    float* __restrict__ h,
    float* __restrict__ c,
    int batch_size,
    int hidden_size
) {
    const int total = batch_size * hidden_size;
    const int stride = blockDim.x * gridDim.x;
    
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < total; 
        idx += stride) {
        const int b = idx / hidden_size;
        const int n = idx % hidden_size;
        const int gate_base = b * hidden_size * 4;

        const float i = sigmoid(gates[gate_base + n]);
        const float f = sigmoid(gates[gate_base + n + hidden_size]);
        const float g = tanhf(gates[gate_base + n + 2*hidden_size]);
        const float o = sigmoid(gates[gate_base + n + 3*hidden_size]);

        const float c_new = f * prev_c[idx] + i * g;
        c[idx] = c_new;
        h[idx] = o * tanhf(c_new);
    }
}

__global__ void linear_stride_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_dim,
    int out_dim,
    int batch_size
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for(int idx = gid; idx < batch_size * out_dim; idx += stride) {
        const int b = idx / out_dim;
        const int o = idx % out_dim;
        
        float sum = 0.0f;
        const float* w_row = &weight[o * in_dim];
        const float* x_row = &input[b * in_dim];
        
        for(int i = 0; i < in_dim; ++i) {
            sum += x_row[i] * w_row[i];
        }
        
        if(bias) sum += bias[o];
        output[idx] = sum;
    }
}

} // namespace

torch::Tensor lstm_forward_stride(
    torch::Tensor input,
    torch::Tensor w_ih,
    torch::Tensor w_hh,
    torch::Tensor b_ih,
    torch::Tensor b_hh,
    torch::Tensor h0,
    torch::Tensor c0
) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = h0.size(1);

    torch::Tensor h = h0.clone();
    torch::Tensor c = c0.clone();
    std::vector<torch::Tensor> outputs;

    constexpr int threads = 256;
    const int elements = batch_size * hidden_size;
    const int blocks = (elements + threads - 1) / threads;

    for(int t = 0; t < seq_len; ++t) {
        torch::Tensor gates = torch::addmm(b_ih, input.select(1, t), w_ih.t())
                              .addmm_(h, w_hh.t())
                              .add_(b_hh);

        lstm_elementwise_stride<<<std::min(blocks, 65535), threads>>>(
            gates.data_ptr<float>(),
            c.data_ptr<float>(),
            h.data_ptr<float>(),
            c.data_ptr<float>(),
            batch_size,
            hidden_size
        );

        outputs.push_back(h.unsqueeze(1));
    }

    return torch::cat(outputs, 1);
}

torch::Tensor linear_forward_stride(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_dim = input.size(1);
    const int out_dim = weight.size(0);
    
    auto output = torch::empty({batch_size, out_dim}, input.options());
    
    constexpr int threads = 256;
    const int elements = batch_size * out_dim;
    const int blocks = (elements + threads - 1) / threads;

    linear_stride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        in_dim,
        out_dim,
        batch_size
    );

    return output;
}

torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> lstm_weights_ih,
    std::vector<torch::Tensor> lstm_weights_hh,
    std::vector<torch::Tensor> lstm_biases_ih,
    std::vector<torch::Tensor> lstm_biases_hh,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias,
    torch::Tensor h0,
    torch::Tensor c0,
    bool is_training
) {
    h0 = h0.to(x.device());
    c0 = c0.to(x.device());

    torch::Tensor out = x;
    const int layers = lstm_weights_ih.size();

    for(int i = 0; i < layers; ++i) {
        out = lstm_forward_stride(
            out,
            lstm_weights_ih[i].to(x.device()),
            lstm_weights_hh[i].to(x.device()),
            lstm_biases_ih[i].to(x.device()),
            lstm_biases_hh[i].to(x.device()),
            h0.narrow(0, i, 1).squeeze(0),
            c0.narrow(0, i, 1).squeeze(0)
        );
    }

    return linear_forward_stride(out.select(1, -1), fc_weight, fc_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-stride optimized LSTM");
}
