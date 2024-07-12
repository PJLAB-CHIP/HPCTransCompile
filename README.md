# LLM4HPSTRansCompile

# Data Preprocess

这部分主要对原始的数据集进行处理，核心执行文件为`prepocess.sh`，可将原始数据进行prompt组装（可选）并进行训练集和测试集的划分。

**Keypoints**

1. 原始数据路径|PROMPT_TEMPLATE的指定

   `utils.py`中修改

   ```python
   class CONFIG:
       # preprocess.py
       raw_data_path = ['./raw_data/raw_data_v2.0_nn.json','./raw_data/raw_data_v2.0_topi.json']
       WITH_IR_INPUT_TEMPLATE = "### IR Code:\n\n{}\n\n### CUDA Code:\n{}\n\n"
       instruction_path = "./instruction/instruction_v1.0.json"
       # format_conversion.py
       codexglue_data_path = '/code/LLM4HPCTransCompile/preprocess/raw_data/hpc_CodeXGLUE_java_cs'
   ```

2. raw data --> data 示例

   **data里面需要包含下面几个项：instruction、input、output、op_name，如果进行端到端的finetune，那么数据集就如下例所示，如果进行instruction tuning，就需要将input改为包含instruction的版本，这个可以在执行preprocess.sh的时候调整超参数**

   ```
   # data
   {
           "instruction": "Suppose your deep learning application is trained in a CUDA-enabled environment. Now, you need to port the application to an environment that doesn't support GPUs. Convert the CUDA operators provided to CPU operators to maintain application functionality in an environment without GPUs.",
           "input": "extern \"C\" __global__ void __launch_bounds__(9) default_function_kernel(float* __restrict__ compute, float* __restrict__ tarray) {\n  compute[((int)threadIdx.x)] = acosf(tarray[((int)threadIdx.x)]);\n}",
           "output": "TVM_DLL int32_t default_function(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);\n#ifdef __cplusplus\nextern \"C\"\n#endif\nTVM_DLL float acosf(float);\n#ifdef __cplusplus\nextern \"C\"\n#endif\nTVM_DLL int32_t default_function(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {\n  int32_t tarray_code = arg_type_ids[0];\n  int32_t compute_code = arg_type_ids[1];\n  void* tarray = (((TVMValue*)args)[0].v_handle);\n  void* compute = (((TVMValue*)args)[1].v_handle);\n  void* tarray_1 = (((DLTensor*)tarray)[0].data);\n  void* default_function_tarray_shape = (((DLTensor*)tarray)[0].shape);\n  void* default_function_tarray_strides = (((DLTensor*)tarray)[0].strides);\n  int32_t dev_id = (((DLTensor*)tarray)[0].device.device_id);\n  void* compute_1 = (((DLTensor*)compute)[0].data);\n  void* default_function_compute_shape = (((DLTensor*)compute)[0].shape);\n  void* default_function_compute_strides = (((DLTensor*)compute)[0].strides);\n  if (!(default_function_tarray_strides == NULL)) {\n  }\n  if (!(default_function_compute_strides == NULL)) {\n  }\n  for (int32_t i0 = 0; i0 < 9; ++i0) {\n    ((float*)compute_1)[i0] = acosf(((float*)tarray_1)[i0]);\n  }\n  return 0;\n}",
           "op_name": "acos"
       }
   ```

- [ ] 需要将op.c和op.cu文件转化到.json中，形成raw_data，详细格式见`preprocess/raw_data`文件夹

# Train

核心执行`finetune.sh`

**核心参数**

- model_name_or_path：模型路径
- output_dir：输出路径
- dataset_name_or_path： 数据集路径
- dataset_format：基本就用input-output 

```
python finetune.py \
    --model_name_or_path /code/model/CodeLlama-13b-hf \
    --output_dir output/hpc_v3.0_topi_without_ir_by_name_simplify \
    --dataset_name_or_path /code/LLM4HPCTransCompile/preprocess/data/hpc_v3.0_topi_without_ir_by_name_simplify/hpc_train_v3.0_topi_without_ir_by_name_simplify.json \
    --dataset_format input-output \
    --use_wandb True \
    --wandb_run_name hpc_v3.0_topi_without_ir_by_name_simplify_3_15 \
    --device_map auto \
    --full_finetune True \
    --source_max_len 2048 \
    --target_max_len 4096 \
    --use_lora True \
```

