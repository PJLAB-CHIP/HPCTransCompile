class CONFIG:
    raw_data_path = ['./raw_data/raw_data_v2.0_nn.json','./raw_data/raw_data_v2.0_topi.json']
    WITH_IR_INPUT_TEMPLATE = "### IR Code:\n\n{}\n\n### CUDA Code:\n{}\n\n"
    instruction_path = "./instruction/instruction_v1.0.json"


def extract_func_implementation(raw_code:str):
    """
    将cpu端算子的函数实现部分抽离出来
    """