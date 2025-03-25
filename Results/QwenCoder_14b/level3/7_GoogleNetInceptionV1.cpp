__global__ void conv2d_optimized_kernel(const float* input, const float* weight,
    const float* bias, float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int stride, int padding) {
const int n = blockIdx.x;
const int oc = blockIdx.y;
const int oh = blockIdx.z * blockDim.y + threadIdx.y;
const int ow = blockIdx.z * blockDim.x + threadIdx.x;

const int output_height = (height + 2 * padding - kernel_size) / stride + 1;
const int output_width = (width + 2 * padding - kernel_size) / stride + 1;

if (oh < output_height && ow < output_width) {
float sum = bias[oc];

for (int ic = 0; ic < in_channels; ++ic) {
for (int kh = 0; kh < kernel_size; ++kh) {
for (int kw = 0; kw < kernel_size; ++kw) {
int h_in = oh * stride - padding + kh;
int w_in = ow * stride - padding + kw;

if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
float input_val = input[((n * in_channels + ic) * height + h_in) * width + w_in];
float weight_val = weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
sum += input_val * weight_val;
}
}
}
}

output[((n * out_channels + oc) * output_height + oh) * output_width + ow] = sum;
}
}