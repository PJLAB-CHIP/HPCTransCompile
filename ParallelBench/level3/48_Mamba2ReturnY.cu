#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor segsum_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);

    int T = x.size(-1);
    
    // Expand x: "... d -> ... d e"
    auto sizes = x.sizes().vec();
    sizes.push_back(T);  // Add an extra dimension
    auto x_expanded = x.unsqueeze(-1).expand(sizes).contiguous();  // [..., T, T]

    // First mask: tril with diagonal=-1
    auto mask1 = torch::tril(torch::ones({T, T}, x.options().dtype(torch::kBool)), -1);
    x_expanded = x_expanded.masked_fill(~mask1, 0);  // Fill outside lower triangle with 0

    // Cumulative sum along dim=-2
    auto x_segsum = torch::cumsum(x_expanded, -2);  // [..., T, T]

    // Second mask: tril with diagonal=0
    auto mask2 = torch::tril(torch::ones({T, T}, x.options().dtype(torch::kBool)), 0);
    x_segsum = x_segsum.masked_fill(~mask2, -std::numeric_limits<float>::infinity());  // Fill outside with -inf

    return x_segsum;
}

// Main forward function (Step 1 + Step 2 with torch::einsum)
torch::Tensor mamba2_forward_cuda(
    torch::Tensor X,
    torch::Tensor initial_states,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int chunk_size) {
    
    CHECK_CUDA(X); CHECK_CUDA(A); CHECK_CUDA(B); CHECK_CUDA(C);
    CHECK_CONTIGUOUS(X); CHECK_CONTIGUOUS(B); CHECK_CONTIGUOUS(C); CHECK_CONTIGUOUS(A);

    int batch_size = A.size(0);    // 16
    int seq_length = A.size(1);    // 128
    int n_heads = A.size(2);       // 8
    int d_state = B.size(3);       // 16
    int d_head = X.size(3);        // 64
    int num_chunks = seq_length / chunk_size;  // 2

    // Step 1: Rearrange A and compute cumsum
    auto A_rearranged = A.view({batch_size, num_chunks, chunk_size, n_heads})
                        .permute({0, 3, 1, 2}).contiguous();  // [16, 8, 2, 64]
    auto max_a = A_rearranged.max(-1, true);
    auto A_normalized = A_rearranged;// - max_a.values;
    auto A_cumsum = A_normalized.cumsum(-1);  // [16, 8, 2, 64]

    // Step 2: Compute L and Y_diag
    auto L = segsum_cuda(A_rearranged).exp();  // [16, 8, 2, 64, 64]

    // Rearrange B, C, X
    auto B_rearranged = B.view({batch_size, num_chunks, chunk_size, n_heads, d_state}).contiguous();  // [16, 2, 64, 8, 16]
    auto C_rearranged = C.view({batch_size, num_chunks, chunk_size, n_heads, d_state}).contiguous();  // [16, 2, 64, 8, 16]
    auto X_rearranged = X.view({batch_size, num_chunks, chunk_size, n_heads, d_head}).contiguous();   // [16, 2, 64, 8, 64]
    auto Y_diag = torch::einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
                               {C_rearranged, B_rearranged, L, X_rearranged});  // [16, 2, 64, 8, 64]

    // Step 3: Compute intra-chunk states
    auto last_A_cumsum = A_cumsum.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), -1})
                        .unsqueeze(-1);  // [16, 8, 2, 1]
    auto decay_states = torch::exp(last_A_cumsum -A_cumsum);  // [16, 8, 2, 64]
    auto states = torch::einsum("bclhn,bhcl,bclhp->bchpn",
                               {B_rearranged, decay_states, X_rearranged});  // [16, 2, 8, 64, 16]
    
    // Step 4: Compute inter-chunk states
    if (!initial_states.defined() || initial_states.numel() == 0) {
        initial_states = torch::zeros_like(states.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}));  // [16, 1, 8, 64, 16]
    }
    states = torch::cat({initial_states, states}, 1);  // [16, 3, 8, 64, 16]
    auto padded_A_cumsum = torch::pad(A_cumsum.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), -1}), {1, 0});  // [16, 8, 3]
    auto decay_chunk = torch::exp(segsum_cuda(padded_A_cumsum));  // [16, 8, 3, 3]
    auto new_states = torch::einsum("bhzc,bchpn->bzhpn", {decay_chunk, states});  // [16, 3, 8, 64, 16]
    states = new_states.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)});  // [16, 2, 8, 64, 16]
    auto final_state = new_states.index({torch::indexing::Slice(), -1});  // [16, 8, 64, 16]

    // Step 5: Compute output
    // Step 5: Compute state-to-output conversion
    auto state_decay_out = torch::exp(A_cumsum);  // [16, 8, 2, 64]
    auto Y_off = torch::einsum("bclhn,bchpn,bhcl->bclhp", {C_rearranged, states, state_decay_out});  // [16, 2, 64, 8, 64]
    auto Y = (Y_diag + Y_off).contiguous().view({batch_size, num_chunks * chunk_size, n_heads, d_head});  // [16, 128, 8, 64]

    // Return intermediate results for debugging
    return Y;
}

// PyBind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mamba2_forward_cuda, "Mamba2 CUDA forward (Step 1 + 2 + 3)");
}
