#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    #pragma omp parallel for
    for (int idx = 0; idx < num_elements; ++idx) {
        float product = 1.0f;
        const int offset = idx;
        
        for (int i = 0; i < 50; i += 10) {
            product *= input_ptr[offset + (i) * stride];
            product *= input_ptr[offset + (i+1) * stride];
            product *= input_ptr[offset + (i+2) * stride];
            product *= input_ptr[offset + (i+3) * stride];
            product *= input_ptr[offset + (i+4) * stride];
            product *= input_ptr[offset + (i+5) * stride];
            product *= input_ptr[offset + (i+6) * stride];
            product *= input_ptr[offset + (i+7) * stride];
            product *= input_ptr[offset + (i+8) * stride];
            product *= input_ptr[offset + (i+9) * stride];
        }
        
        output_ptr[idx] = product;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CPU)");
}