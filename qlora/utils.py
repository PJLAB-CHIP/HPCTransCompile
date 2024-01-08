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


def show_model_attribute(model):
    """
    显示模型属性
    """
    # print('model:', model)
    """
    LlamaForCausalLM(
    (model): LlamaModel(
        (embed_tokens): Embedding(32016, 5120)
        (layers): ModuleList(
        (0-39): 40 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
            (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
        )
        )
        (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=5120, out_features=32016, bias=False)  # needed for 16-bit
    )
    """
    # print('model.state_dict().keys():', model.state_dict().keys())
    print('model parameter matrices num:', len(model.state_dict().keys()))
    """
    1. embedding层
    'model.embed_tokens.weight' torch.Size([32016, 5120])
    2. transformer层 0-39
    'model.layers.0.self_attn.q_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.self_attn.k_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.self_attn.v_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.self_attn.o_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.mlp.gate_proj.weight', torch.Size([13824, 5120])
    'model.layers.0.mlp.up_proj.weight', torch.Size([13824, 5120])
    'model.layers.0.mlp.down_proj.weight', torch.Size([5120, 13824])
    'model.layers.0.input_layernorm.weight', torch.Size([5120])
    'model.layers.0.post_attention_layernorm.weight' torch.Size([5120])
    3. norm层
    'model.norm.weight', torch.Size([5120])
    'lm_head.weight' torch.Size([32016, 5120])
    """