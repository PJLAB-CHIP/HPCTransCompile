#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Function to compute partial sums for the channel
void computePartialSums(const float* input, int c, int N, int C, int H, int W, float& partialSum, float& partialSumSq) {
    int numElements = N * H * W;
    partialSum = 0.0f;
    partialSumSq = 0.0f;
    for (int i = 0; i < numElements; ++i) {
        int n = i / (H * W);
        int r = i % (H * W);
        int h = r / W;
        int w = r % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        partialSum += val;
        partialSumSq += val * val;
    }
}

// Function to normalize a value
float normalizeValue(float val, float mean, float invStd, float w, float b) {
    return (val - mean) * invStd * w + b;
}

// Host function called from PyTorch
torch::Tensor adaptive_forward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        float partialSum, partialSumSq;
        computePartialSums(input.data_ptr<float>(), c, N, C, H, W, partialSum, partialSumSq);

        float mean = partialSum / (N * H * W);
        float var = partialSumSq / (N * H * W) - mean * mean;

        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }

        float invStd = std::sqrt(1.0f / (var + eps));
        float channelWeight = weight[c].item<float>();
        float channelBias = bias[c].item<float>();

        #pragma omp parallel for
        for (int i = 0; i < N * H * W; ++i) {
            int n = i / (H * W);
            int r = i % (H * W);
            int h = r / W;
            int w = r % W;
            int idx = ((n * C + c) * H + h) * W + w;
            float val = input[idx];
            output[idx] = normalizeValue(val, mean, invStd, channelWeight, channelBias);
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_forward_cpu, "Adaptive Block Size BatchNorm forward (CPU)");
}
