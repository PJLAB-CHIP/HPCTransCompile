#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define NUM_STREAMS 4

// This kernel optimizes workload distribution by ensuring each thread block
// processes a balanced number of timesteps, reducing idle threads and improving
// overall GPU utilization.

__global__ void lstm_balanced_workload_kernel(
    const float* __restrict__ x,           // [batch_size, total_seq_len, input_size]
    float* __restrict__ y,                 // [batch_size, total_seq_len, hidden_size]
    float* __restrict__ h,                 // [batch_size, hidden_size]
    float* __restrict__ c,                 // [batch_size, hidden_size]
    const float* __restrict__ W_ih,        // [4 * hidden_size, input_size]
    const float* __restrict__ W_hh,        // [4 * hidden_size, hidden_size]
    const float* __restrict__ bias_ih,     // [4 * hidden_size]
    const float* __restrict__ bias_hh,     // [4 * hidden_size]
    const int seq_start,                   // Starting timestep index
    const int seq_length,                  // Number of timesteps to process in this kernel launch
    const int total_seq_len,               // Total number of timesteps
    const int input_size,
    const int hidden_size
) {
    // Each block processes one batch sample
    int batch = blockIdx.x;
    int tid = threadIdx.x;  // Assumed to be in [0, hidden_size)

    // Pointers for the batch
    const float* x_batch = x + batch * total_seq_len * input_size;
    float* y_batch = y + batch * total_seq_len * hidden_size;
    float* h_ptr = h + batch * hidden_size;
    float* c_ptr = c + batch * hidden_size;

    // Each thread loads its corresponding hidden and cell state
    float h_val = h_ptr[tid];
    float c_val = c_ptr[tid];

    // Shared memory to store the complete hidden state of the previous timestep
    extern __shared__ float s_hidden[]; // size = hidden_size * sizeof(float)

    // Process each timestep in the assigned chunk
    for (int t = seq_start; t < seq_start + seq_length; t++) {
        // Write current hidden state into shared memory for use by all threads
        s_hidden[tid] = h_val;
        __syncthreads();

        int offset = tid;  // Each thread handles one specific hidden unit

        // Initialize gates with biases using __ldg() for read-only access
        float i_gate = __ldg(&bias_ih[offset]) + __ldg(&bias_hh[offset]);
        float f_gate = __ldg(&bias_ih[hidden_size + offset]) + __ldg(&bias_hh[hidden_size + offset]);
        float g_gate = __ldg(&bias_ih[2 * hidden_size + offset]) + __ldg(&bias_hh[2 * hidden_size + offset]);
        float o_gate = __ldg(&bias_ih[3 * hidden_size + offset]) + __ldg(&bias_hh[3 * hidden_size + offset]);

        // Input contribution: process input vector with 128-bit (float4) aligned vectorized loads
        const float* x_t = x_batch + t * input_size;
        int k_vectorized = (input_size / 4) * 4;  // Largest multiple of 4 less than or equal to input_size
        const float4* x_t_vec = reinterpret_cast<const float4*>(x_t);
        int num_vec = k_vectorized / 4;

        for (int k = 0; k < num_vec; k++) {
            float4 x_val = __ldg(&x_t_vec[k]);
            // For each gate, compute the dot product using vectorized weights
            const float4* w_i = reinterpret_cast<const float4*>(W_ih + (0 * hidden_size + offset) * input_size);
            const float4* w_f = reinterpret_cast<const float4*>(W_ih + (1 * hidden_size + offset) * input_size);
            const float4* w_g = reinterpret_cast<const float4*>(W_ih + (2 * hidden_size + offset) * input_size);
            const float4* w_o = reinterpret_cast<const float4*>(W_ih + (3 * hidden_size + offset) * input_size);
            float4 w_i_val = __ldg(&w_i[k]);
            float4 w_f_val = __ldg(&w_f[k]);
            float4 w_g_val = __ldg(&w_g[k]);
            float4 w_o_val = __ldg(&w_o[k]);

            i_gate += x_val.x * w_i_val.x + x_val.y * w_i_val.y + x_val.z * w_i_val.z + x_val.w * w_i_val.w;
            f_gate += x_val.x * w_f_val.x + x_val.y * w_f_val.y + x_val.z * w_f_val.z + x_val.w * w_f_val.w;
            g_gate += x_val.x * w_g_val.x + x_val.y * w_g_val.y + x_val.z * w_g_val.z + x_val.w * w_g_val.w;
            o_gate += x_val.x * w_o_val.x + x_val.y * w_o_val.y + x_val.z * w_o_val.z + x_val.w * w_o_val.w;
        }
        // Process any remaining input elements
        for (int k = k_vectorized; k < input_size; k++) {
            float x_val = __ldg(&x_t[k]);
            i_gate += x_val * __ldg(&W_ih[(0 * hidden_size + offset) * input_size + k]);
            f_gate += x_val * __ldg(&W_ih[(1 * hidden_size + offset) * input_size + k]);
            g_gate += x_val * __ldg(&W_ih[(2 * hidden_size + offset) * input_size + k]);
            o_gate += x_val * __ldg(&W_ih[(3 * hidden_size + offset) * input_size + k]);
        }

        // Hidden state contribution from previous time step
        int hidden_vectorized = (hidden_size / 4) * 4;
        const float* shared_ptr = s_hidden;
        const float4* shared_vec = reinterpret_cast<const float4*>(shared_ptr);
        int num_hidden_vec = hidden_vectorized / 4;
        for (int k = 0; k < num_hidden_vec; k++) {
            float4 h_vec = shared_vec[k];
            const float4* wh_i = reinterpret_cast<const float4*>(W_hh + (0 * hidden_size + offset) * hidden_size);
            const float4* wh_f = reinterpret_cast<const float4*>(W_hh + (1 * hidden_size + offset) * hidden_size);
            const float4* wh_g = reinterpret_cast<const float4*>(W_hh + (2 * hidden_size + offset) * hidden_size);
            const float4* wh_o = reinterpret_cast<const float4*>(W_hh + (3 * hidden_size + offset) * hidden_size);
            float4 wh_i_val = __ldg(&wh_i[k]);
            float4 wh_f_val = __ldg(&wh_f[k]);
            float4 wh_g_val = __ldg(&wh_g[k]);
            float4 wh_o_val = __ldg(&wh_o[k]);

            i_gate += h_vec.x * wh_i_val.x + h_vec.y * wh_i_val.y + h_vec.z * wh_i_val.z + h_vec.w * wh_i_val.w;
            f_gate += h_vec.x * wh_f_val.x + h_vec.y * wh_f_val.y + h_vec.z * wh_f_val.z + h_vec.w * wh_f_val.w;
            g_gate += h_vec.x * wh_g_val.x + h_vec.y * wh_g_val.y + h_vec.z * wh_g_val.z + h_vec.w * wh_g_val.w;
            o_gate += h_vec.x * wh_o_val.x + h_vec.y * wh_o_val.y + h_vec.z * wh_o_val.z + h_vec.w * wh_o_val.w;
        }
        for (int k = hidden_vectorized; k < hidden_size; k++) {
            float h_shared = s_hidden[k];
            i_gate += h_shared * __ldg(&W_hh[(0 * hidden_size + offset) * hidden_size + k]);
            f_gate += h_shared * __ldg(&W_hh[(1 * hidden_size + offset) * hidden_size + k]);
            g_gate += h_shared * __ldg(&W_hh[(2 * hidden_size + offset) * hidden_size + k]);
            o_gate += h_shared * __ldg(&W_hh[(3 * hidden_size + offset) * hidden_size + k]);
        }

        // Apply activation functions
        i_gate = 1.0f / (1.0f + expf(-i_gate));
        f_gate = 1.0f / (1.0f + expf(-f_gate));
        g_gate = tanhf(g_gate);
        o_gate = 1.0f / (1.0f + expf(-o_gate));

        // Update cell state and hidden state
        c_val = f_gate * c_val + i_gate * g_gate;
        h_val = o_gate * tanhf(c_val);

        // Write the output for this timestep
        y_batch[t * hidden_size + offset] = h_val;
        __syncthreads();
    }

    // If this kernel processed the final chunk, write back the final states
    if (seq_start + seq_length == total_seq_len) {
        h_ptr[tid] = h_val;
        c_ptr[tid] = c_val;
    }
}


torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> lstm_weights_ih,
    std::vector<torch::Tensor> lstm_weights_hh,
    std::vector<torch::Tensor> lstm_biases_ih,
    std::vector<torch::Tensor> lstm_biases_hh,
    torch::Tensor h0,
    torch::Tensor c0,
    bool is_training
) {
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Ensure h0 and c0 are on the same device as x
    h0 = h0.to(x.device());
    c0 = c0.to(x.device());

    const int64_t num_layers = lstm_weights_ih.size();
    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t input_size = x.size(2);
    const int64_t hidden_size = h0.size(2);

    // Determine chunk size for processing the sequence in parallel streams
    const int chunk_size = (seq_len + NUM_STREAMS - 1) / NUM_STREAMS;
    torch::Tensor out = x;

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        auto weight_ih = lstm_weights_ih[layer];
        auto weight_hh = lstm_weights_hh[layer];
        auto bias_ih = lstm_biases_ih[layer];
        auto bias_hh = lstm_biases_hh[layer];

        torch::Tensor h_layer = h0.select(0, layer);
        torch::Tensor c_layer = c0.select(0, layer);

        auto layer_out = torch::empty({batch_size, seq_len, hidden_size}, x.options());

        dim3 grid(batch_size);
        dim3 block(hidden_size); // each thread corresponds to one hidden element
        size_t shared_mem = hidden_size * sizeof(float);

        // Launch sequence chunks in separate CUDA streams
        for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
            int seq_start_idx = stream_idx * chunk_size;
            int seq_chunk = std::min(chunk_size, static_cast<int>(seq_len - seq_start_idx));
            if (seq_chunk <= 0) continue;
            cudaStreamSynchronize(streams[stream_idx]);
            lstm_balanced_workload_kernel<<<grid, block, shared_mem, streams[stream_idx]>>>(
                out.data_ptr<float>(),
                layer_out.data_ptr<float>(),
                h_layer.data_ptr<float>(),
                c_layer.data_ptr<float>(),
                weight_ih.data_ptr<float>(),
                weight_hh.data_ptr<float>(),
                bias_ih.data_ptr<float>(),
                bias_hh.data_ptr<float>(),
                seq_start_idx,
                seq_chunk,
                seq_len,
                (layer == 0) ? input_size : hidden_size,
                hidden_size
            );
        }

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        out = layer_out;
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return c0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LSTM forward with balanced workload distribution");
}
