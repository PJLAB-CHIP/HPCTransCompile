#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets)
{
    // Ensure inputs are on CPU
    TORCH_CHECK(predictions.is_cuda() == false, "predictions must be a CPU tensor");
    TORCH_CHECK(targets.is_cuda() == false, "targets must be a CPU tensor");

    // Ensure inputs have correct dimensions
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    // Ensure data types are correct
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    // Output tensor for losses per sample
    auto losses = torch::empty({batch_size}, predictions.options());

    // Parallel loop over the batch
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++)
    {
        // Get pointer to logits for sample i
        const float* logits_i = predictions.data_ptr<float>() + i * num_classes;
        int64_t target = targets.data_ptr<int64_t>()[i];

        // Compute max logit for numerical stability
        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; j++)
        {
            if (logits_i[j] > max_logit)
                max_logit = logits_i[j];
        }

        // Compute sum of exp(logits - max_logit)
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++)
        {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        // Compute log_sum_exp
        float log_sum_exp = logf(sum_exp);

        // Compute loss for this sample
        float loss = - (logits_i[target] - max_logit - log_sum_exp);
        losses[i] = loss;
    }

    // Compute mean loss over batch
    auto loss = losses.mean();

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward (CPU)");
}