from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    AutoModel,
    pipeline
)

def count_tokens(text, model_name):
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 对文本进行tokenization
    tokens = tokenizer.tokenize(text)

    # 打印token数目
    print("Token 数目:", len(tokens))
    print("Tokens:", tokens)

# 要处理的文本
text_to_tokenize = "### IR Code:\n\n# from tvm.script import ir as I\n# from tvm.script import tir as T\n\n@I.ir_module\nclass Module:\n    @T.prim_func\n    def main(data: T.Buffer((5, 24, 28, 8), \"float32\"), compute: T.Buffer((5, 24, 28, 8), \"float32\")):\n        T.func_attr({\"from_legacy_te_schedule\": T.bool(True), \"tir.noalias\": T.bool(True)})\n        for i0_i1_fused in T.parallel(120):\n            for i2, i3_s in T.grid(28, 8):\n                cse_var_1: T.int32 = i0_i1_fused * 224 + i2 * 8 + i3_s\n                compute_1 = T.Buffer((26880,), data=compute.data)\n                data_1 = T.Buffer((26880,), data=data.data)\n                compute_1[cse_var_1] = T.acos(data_1[cse_var_1])\n\n### CUDA Code:\n\n#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \\\n     (__CUDACC_VER_MAJOR__ > 11))\n#define TVM_ENABLE_L2_PREFETCH 1\n#else\n#define TVM_ENABLE_L2_PREFETCH 0\n#endif\n\n#ifdef _WIN32\n  using uint = unsigned int;\n  using uchar = unsigned char;\n  using ushort = unsigned short;\n  using int64_t = long long;\n  using uint64_t = unsigned long long;\n#else\n  #define uint unsigned int\n  #define uchar unsigned char\n  #define ushort unsigned short\n  #define int64_t long long\n  #define uint64_t unsigned long long\n#endif\nextern \"C\" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ compute, float* __restrict__ data);\nextern \"C\" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {\n  compute[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = acosf(data[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))]);\n}\n\n\n\n"

# 预训练模型的名称，例如 'bert-base-uncased'
model_name = '/code/model/CodeLlama-34b-Instruct-hf'

# 计算token数目
count_tokens(text_to_tokenize, model_name)
