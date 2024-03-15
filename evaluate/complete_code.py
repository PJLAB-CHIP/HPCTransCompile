import json
import os
from os.path import join,exists

FOLDER_NAME = 'CodeLlama-13b-hf-400-v3.0_topi_without_ir_by_name_simplify-v3.0_topi_without_ir_by_name_simplify_3_12'
PREFIX = '// tvm target: c -keys=cpu \n#define TVM_EXPORTS\n#include \"tvm/runtime/c_runtime_api.h\"\n#include \"tvm/runtime/c_backend_api.h\"\n#include <math.h>\n#include <stdbool.h>\n#ifdef __cplusplus\nextern \"C\"\n#endif\n'
SUFFIX = '\n\n// CodegenC: NOTE: Auto-generated entry function\n#ifdef __cplusplus\nextern \"C\"\n#endif\nTVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {\n  return default_function(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);\n}\n'

if __name__ == '__main__':
    file_list = os.listdir(join(FOLDER_NAME,'generate'))
    for file_name in file_list:

        # step1 add prefix and suffix
        with open(join(FOLDER_NAME,'generate',file_name),'r') as file:
            c_code = file.read()
            c_code = PREFIX+c_code+SUFFIX
        with open(join(FOLDER_NAME,'generate',file_name),'w') as file:
            file.write(c_code)

        # (optional) remove comma
        # with open(join(FOLDER_NAME,'generate',file_name),'r') as file:
        #     c_code = file.read()
        #     c_code = c_code.replace('},','}')
        # with open(join(FOLDER_NAME,'generate',file_name),'w') as file:
        #     file.write(c_code)