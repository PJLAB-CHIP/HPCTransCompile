ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def show_model_info(model):
    for name, module in model.named_modules():
        print('name:', name)
        print('type(module):', type(module))

    for param in model.parameters():
        print(f"Parameter: {param.shape}, Data Type: {param.dtype}")

def extract_alpaca_dataset(example):
    if example.get('input','') != '':
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def cal_model_params(model):
    """
    计算模型的参数量
    """
    model_param_num = 0
    layer_list = list(model.state_dict().keys())
    for layer in layer_list:
        layer_matrix = model.state_dict()[layer]
        layer_param_num = 1
        for i in range(layer_matrix.ndim):
            layer_param_num *= layer_matrix.shape[i]
        model_param_num += layer_param_num
    print('model_param_num:', model_param_num)
    return model_param_num