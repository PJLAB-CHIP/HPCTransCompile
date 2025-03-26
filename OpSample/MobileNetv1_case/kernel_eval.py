import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def load_ref_src(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable, callable]: # type: ignore
    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None # type: ignore

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None # type: ignore

    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    module_fn = context.get("module_fn")
    return (Model, get_init_inputs_fn, get_inputs_fn, module_fn) # type: ignore

def load_cuda_module(kernel_path):
    task_name = kernel_path.split("/")[-1].split(".")[0]
    task_name = "_".join(task_name.split("_")[1:])   # Remove problem ID
    if task_name == "":
        task_name = "task"

    print("@@@@@---> ", task_name, kernel_path)
    cuda_module = load(
        name=task_name,
        sources=[kernel_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

if __name__ == "__main__":
    cuda_path = '/code/LLM4HPCTransCompile/OpSample/MobileNetv1_case/19_MobileNetV1_functional.cu'
    functional_path = "/code/LLM4HPCTransCompile/OpSample/MobileNetv1_case/19_MobileNetV1_functional.py"

    ref_src = load_ref_src(functional_path)
    cuda_module = load_cuda_module(cuda_path)

    context = {}
    Model, get_init_inputs, get_inputs, module_fn = load_original_model_and_inputs(ref_src, context)
    init_inputs = get_init_inputs()
    init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]

    with torch.no_grad():
        model = Model(*init_inputs)
        assert hasattr(model, "forward")

    with torch.no_grad():
        inputs = get_inputs()
        inputs = [ x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]        
        model = model.cuda()

        # get ref output
        ref_out = model(*inputs, fn=module_fn)
        print("ref_out:\n", ref_out)

        # replace with cuda module forward kernel
        new_out = model(*inputs, fn=cuda_module.forward)
        print("new_out:\n", new_out)


        if not torch.allclose(ref_out, new_out, atol=1e-02, rtol=1e-02):
            print("!!!!!!! Compare Fail\n")
        else:
            print("!!!!!!! Compare Pass\n")