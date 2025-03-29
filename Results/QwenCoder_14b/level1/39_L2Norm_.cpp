#include <torch/extension.h>
#include <omp.h>
#include <cmath>

template <typename scalar_t>
void l2norm_strided_cpu(
    const scalar_t* input,
    scalar_t* output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {

    #pragma omp parallel for
    for (int vector_idx = 0; vector_idx < total_vectors; ++vector_idx) {
        const int base = vector_idx * outer_stride;
        scalar_t thread_sum = 0;

        if (stride_C == 1) {
            const int vec_size = sizeof(scalar_t) == 4 ? 4 : 2;
            const int aligned_C = (C / vec_size) * vec_size;

            if constexpr (sizeof(scalar_t) == 4) {
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                const int num_vectors = aligned_C / 4;

                for (int i = 0; i < num_vectors; ++i) {
                    float4 v = in_vec[i];
                    thread_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                }
            } else {
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                const int num_vectors = aligned_C / 2;

                for (int i = 0; i < num_vectors; ++i) {
                    double2 v = in_vec[i];
                    thread_sum += v.x * v.x + v.y * v.y;
                }
            }

            for (int i = aligned_C; i < C; ++i) {
                scalar_t val = input[base + i];
                thread_sum += val * val;
            }
        } else {
            for (int i = 0; i < C; ++i) {
                scalar_t val = input[base + i * stride_C];
                thread_sum += val * val;
            }
        }

        scalar_t inv_norm = 1.0 / std::sqrt(thread_sum + 1e-12);

        if (stride_C == 1) {
            const int vec_size = sizeof(scalar_t) == 4 ? 4 : 2;
            const int aligned_C = (C / vec_size) * vec_size;

            if constexpr (sizeof(scalar_t) == 4) {
                float4* out_vec = reinterpret_cast<float4*>(output + base);
                const float4* in_vec = reinterpret_cast<const float4*>(input + base);
                const int num_vectors = aligned_C / 4;

                for (int i = 0; i < num_vectors; ++i) {
                    float4 v = in_vec[i];
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                    v.z *= inv_norm;
                    v.w *= inv_norm;
                    out_vec[i] = v;
                }
            } else {
                double2* out_vec = reinterpret_cast<double2*>(output + base);
                const double2* in_vec = reinterpret_cast<const double2*>(input + base);
                const int num_vectors = aligned_C / 2;

                for (int i = 0; i < num_vectors; ++i) {
                    double2 v = in_vec[i];
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                    out_vec[i] = v;
                }
            }

            for (int i = aligned_C; i < C; ++i) {
                output[base + i] = input[base + i] * inv_norm;
            }
        } else {
            for (int i = 0; i < C; ++i) {
                output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(!input.is_cuda(), "Input must be a CPU tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_strided", ([&] {
        l2norm_strided_cpu<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with stride optimization");
}
