#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for patch embedding (unfold + linear)
template <typename scalar_t>
__global__ void patch_embedding_kernel(
    const scalar_t* __restrict__ img,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int img_height,
    int img_width,
    int patch_size,
    int dim,
    int num_patches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_patches * dim) return;

    int b = idx / (num_patches * dim);        // batch index
    int p = (idx / dim) % num_patches;        // patch index
    int d = idx % dim;                        // dimension index

    int ph = p / (img_width / patch_size);    // patch height index
    int pw = p % (img_width / patch_size);    // patch width index
    int patch_dim = channels * patch_size * patch_size;

    scalar_t value = bias[d];
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < patch_size; h++) {
            for (int w = 0; w < patch_size; w++) {
                int img_h = ph * patch_size + h;
                int img_w = pw * patch_size + w;
                if (img_h < img_height && img_w < img_width) {
                    scalar_t pixel = img[b * channels * img_height * img_width +
                                        c * img_height * img_width +
                                        img_h * img_width + img_w];
                    scalar_t wgt = weight[d * patch_dim + c * patch_size * patch_size + h * patch_size + w];
                    value += pixel * wgt;
                }
            }
        }
    }
    output[b * num_patches * dim + p * dim + d] = value;
}

// CUDA kernel for adding cls_token and pos_embedding
template <typename scalar_t>
__global__ void add_cls_pos_kernel(
    const scalar_t* __restrict__ patches,
    const scalar_t* __restrict__ cls_token,
    const scalar_t* __restrict__ pos_embedding,
    scalar_t* __restrict__ output,
    int batch_size,
    int num_patches,
    int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * (num_patches + 1) * dim) return;

    int b = idx / ((num_patches + 1) * dim);  // batch index
    int t = (idx / dim) % (num_patches + 1);  // token index (0 for cls, 1+ for patches)
    int d = idx % dim;                        // dimension index

    if (t == 0) {
        output[idx] = cls_token[d] + pos_embedding[t * dim + d];
    } else {
        output[idx] = patches[b * num_patches * dim + (t - 1) * dim + d] + pos_embedding[t * dim + d];
    }
}

// Full forward pass
torch::Tensor forward(
    torch::Tensor img,
    int patch_size,
    torch::Tensor pos_embedding,
    torch::Tensor patch_to_embedding_weight,
    torch::Tensor patch_to_embedding_bias,
    torch::Tensor cls_token,
    float dropout_p,
    // Transformer layer parameters (6 layers as per depth=6)
    torch::Tensor attn_qkv_weight_0, torch::Tensor attn_qkv_bias_0, torch::Tensor attn_proj_weight_0, torch::Tensor attn_proj_bias_0,
    torch::Tensor linear1_weight_0, torch::Tensor linear1_bias_0, torch::Tensor linear2_weight_0, torch::Tensor linear2_bias_0,
    torch::Tensor norm1_weight_0, torch::Tensor norm1_bias_0, torch::Tensor norm2_weight_0, torch::Tensor norm2_bias_0,
    torch::Tensor attn_qkv_weight_1, torch::Tensor attn_qkv_bias_1, torch::Tensor attn_proj_weight_1, torch::Tensor attn_proj_bias_1,
    torch::Tensor linear1_weight_1, torch::Tensor linear1_bias_1, torch::Tensor linear2_weight_1, torch::Tensor linear2_bias_1,
    torch::Tensor norm1_weight_1, torch::Tensor norm1_bias_1, torch::Tensor norm2_weight_1, torch::Tensor norm2_bias_1,
    torch::Tensor attn_qkv_weight_2, torch::Tensor attn_qkv_bias_2, torch::Tensor attn_proj_weight_2, torch::Tensor attn_proj_bias_2,
    torch::Tensor linear1_weight_2, torch::Tensor linear1_bias_2, torch::Tensor linear2_weight_2, torch::Tensor linear2_bias_2,
    torch::Tensor norm1_weight_2, torch::Tensor norm1_bias_2, torch::Tensor norm2_weight_2, torch::Tensor norm2_bias_2,
    torch::Tensor attn_qkv_weight_3, torch::Tensor attn_qkv_bias_3, torch::Tensor attn_proj_weight_3, torch::Tensor attn_proj_bias_3,
    torch::Tensor linear1_weight_3, torch::Tensor linear1_bias_3, torch::Tensor linear2_weight_3, torch::Tensor linear2_bias_3,
    torch::Tensor norm1_weight_3, torch::Tensor norm1_bias_3, torch::Tensor norm2_weight_3, torch::Tensor norm2_bias_3,
    torch::Tensor attn_qkv_weight_4, torch::Tensor attn_qkv_bias_4, torch::Tensor attn_proj_weight_4, torch::Tensor attn_proj_bias_4,
    torch::Tensor linear1_weight_4, torch::Tensor linear1_bias_4, torch::Tensor linear2_weight_4, torch::Tensor linear2_bias_4,
    torch::Tensor norm1_weight_4, torch::Tensor norm1_bias_4, torch::Tensor norm2_weight_4, torch::Tensor norm2_bias_4,
    torch::Tensor attn_qkv_weight_5, torch::Tensor attn_qkv_bias_5, torch::Tensor attn_proj_weight_5, torch::Tensor attn_proj_bias_5,
    torch::Tensor linear1_weight_5, torch::Tensor linear1_bias_5, torch::Tensor linear2_weight_5, torch::Tensor linear2_bias_5,
    torch::Tensor norm1_weight_5, torch::Tensor norm1_bias_5, torch::Tensor norm2_weight_5, torch::Tensor norm2_bias_5,
    int dim, int heads,
    torch::Tensor mlp_head_0_weight,
    torch::Tensor mlp_head_0_bias,
    torch::Tensor mlp_head_3_weight,
    torch::Tensor mlp_head_3_bias) {

    TORCH_CHECK(img.is_cuda(), "Input must be a CUDA tensor");
    const int batch_size = img.size(0);
    const int channels = img.size(1);
    const int img_height = img.size(2);
    const int img_width = img.size(3);
    const int num_patches = (img_height / patch_size) * (img_width / patch_size);

    // Patch embedding
    auto patches = torch::zeros({batch_size, num_patches, dim}, img.options());
    const int threads = 256;
    int blocks = (batch_size * num_patches * dim + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "patch_embedding", ([&] {
        patch_embedding_kernel<scalar_t><<<blocks, threads>>>(
            img.data_ptr<scalar_t>(),
            patch_to_embedding_weight.data_ptr<scalar_t>(),
            patch_to_embedding_bias.data_ptr<scalar_t>(),
            patches.data_ptr<scalar_t>(),
            batch_size, channels, img_height, img_width, patch_size, dim, num_patches);
    }));
    cudaDeviceSynchronize();

    // Add cls_token and pos_embedding
    auto x = torch::zeros({batch_size, num_patches + 1, dim}, img.options());
    blocks = (batch_size * (num_patches + 1) * dim + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(patches.scalar_type(), "add_cls_pos", ([&] {
        add_cls_pos_kernel<scalar_t><<<blocks, threads>>>(
            patches.data_ptr<scalar_t>(),
            cls_token.data_ptr<scalar_t>(),
            pos_embedding.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            batch_size, num_patches, dim);
    }));
    cudaDeviceSynchronize();

    // Transformer layers
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> layers = {
        {attn_qkv_weight_0, attn_qkv_bias_0, attn_proj_weight_0, attn_proj_bias_0, linear1_weight_0, linear1_bias_0, linear2_weight_0, linear2_bias_0, norm1_weight_0, norm1_bias_0, norm2_weight_0, norm2_bias_0},
        {attn_qkv_weight_1, attn_qkv_bias_1, attn_proj_weight_1, attn_proj_bias_1, linear1_weight_1, linear1_bias_1, linear2_weight_1, linear2_bias_1, norm1_weight_1, norm1_bias_1, norm2_weight_1, norm2_bias_1},
        {attn_qkv_weight_2, attn_qkv_bias_2, attn_proj_weight_2, attn_proj_bias_2, linear1_weight_2, linear1_bias_2, linear2_weight_2, linear2_bias_2, norm1_weight_2, norm1_bias_2, norm2_weight_2, norm2_bias_2},
        {attn_qkv_weight_3, attn_qkv_bias_3, attn_proj_weight_3, attn_proj_bias_3, linear1_weight_3, linear1_bias_3, linear2_weight_3, linear2_bias_3, norm1_weight_3, norm1_bias_3, norm2_weight_3, norm2_bias_3},
        {attn_qkv_weight_4, attn_qkv_bias_4, attn_proj_weight_4, attn_proj_bias_4, linear1_weight_4, linear1_bias_4, linear2_weight_4, linear2_bias_4, norm1_weight_4, norm1_bias_4, norm2_weight_4, norm2_bias_4},
        {attn_qkv_weight_5, attn_qkv_bias_5, attn_proj_weight_5, attn_proj_bias_5, linear1_weight_5, linear1_bias_5, linear2_weight_5, linear2_bias_5, norm1_weight_5, norm1_bias_5, norm2_weight_5, norm2_bias_5}
    };

    for (const auto& layer : layers) {
        auto attn = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(dim, heads));
        attn->in_proj_weight = std::get<0>(layer);
        attn->in_proj_bias = std::get<1>(layer);
        attn->out_proj->weight = std::get<2>(layer);
        attn->out_proj->bias = std::get<3>(layer);
        torch::Tensor empty_tensor = torch::Tensor(); // 空的 Tensor 表示无掩码
        auto attn_out = attn->forward(x, x, x, empty_tensor, false, empty_tensor, true);
        x = x + std::get<0>(attn_out);
        x = torch::layer_norm(x, {dim}, std::get<8>(layer), std::get<9>(layer), 1e-5, true);
        auto ff = torch::linear(x, std::get<4>(layer), std::get<5>(layer));
        ff = torch::gelu(ff);
        ff = torch::linear(ff, std::get<6>(layer), std::get<7>(layer));
        x = x + ff;
        x = torch::layer_norm(x, {dim}, std::get<10>(layer), std::get<11>(layer), 1e-5, true);
    }

    // Extract cls_token and MLP head
    x = x.index({torch::indexing::Slice(), 0}); // [batch_size, dim]
    x = torch::linear(x, mlp_head_0_weight, mlp_head_0_bias);
    x = torch::gelu(x);
    x = torch::linear(x, mlp_head_3_weight, mlp_head_3_bias);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transformer Forward (CUDA)");
}