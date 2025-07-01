def search_op_info(op_info_dict,file_name):
    matching_items = []
    print('file_name:', file_name)
    for item in op_info_dict:
        current_file_name = f"{item['op_name']}_{item['op_args']}_{item['input_shape']}"
        if current_file_name == file_name:
            matching_items.append(item)
    if len(matching_items) != 1:
        raise ValueError('Error: Multiple Matches!')
    return matching_items[0]


def split_string(input_string):
    # 找到最后两个下划线的位置
    first_split = input_string.rfind('_')
    second_split = input_string[:first_split].rfind('_')

    # 根据位置切割字符串
    part1 = input_string[:second_split]
    part2 = input_string[second_split + 1:first_split]
    part3 = input_string[first_split + 1:]

    return part1, part2, part3

def get_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        c_file_content = file.read()
    return c_file_content