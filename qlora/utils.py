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
    # print('model_param_num:', model_param_num)
    return model_param_num