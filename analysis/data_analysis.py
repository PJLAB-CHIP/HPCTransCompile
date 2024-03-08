import random
import seaborn as sns
import transformers
import json
from transformers import(
    LlamaTokenizer,
    CodeLlamaTokenizer
)
import matplotlib.pyplot as plt

class CONFIG:
    data_path = '/code/LLM4HPCTransCompile/preprocess/data/hpc_v3.0_topi_without_ir_by_name_simplify/hpc_data_v3.0_topi_without_ir_by_name_simplify.json'
    model_path = '/code/model/CodeLlama-13b-hf'

def data_fix():
    with open(CONFIG.data_path,'rb') as file:
        data = json.load(file)

    # step1
    # for item in data:
    #     item['input'] = item['input'][:item['input'].rfind(',')] + '\n'
    #     item['output'] = item['output'][:item['output'].rfind(',')] + '\n'

    # step2
    # for item in data:
    #     last_comma_index = item['output'].rfind("return 0;\n}") + len("return 0;\n}")
    #     # print(last_comma_index)
    #     # print(item['output'][last_comma_index])
    #     item['output'] = item['output'][:last_comma_index] + item['output'][last_comma_index+1:]

    for item in data:
        item['input'] = item['input'].replace('},','}')
        item['output'] = item['output'].replace('},','}')
    
    with open(CONFIG.data_path,'w') as file:
        json.dump(data,file,indent=4)

if __name__ == '__main__':
    # data_fix()

    with open(CONFIG.data_path,'rb') as file:
        data = json.load(file)
    tokenier =  CodeLlamaTokenizer.from_pretrained(CONFIG.model_path)
    input_list = []
    output_list = []
    for item in data:
        input_ids = tokenier(item['input'],return_tensors='pt')
        output_ids = tokenier(item['output'],return_tensors='pt')
        input_list.extend(input_ids['input_ids'])
        output_list.extend(output_ids['input_ids'])
        if len(input_ids['input_ids'][0])>1000:
            print(item['input'])

    input_length_list = []
    output_length_list = []
    for item in input_list:
        input_length_list.append(len(item))
    for item in output_list:
        output_length_list.append(len(item))
    
    print(max(input_length_list),min(input_length_list))
    print(max(output_length_list),min(output_length_list))

    # sns.countplot(x=output_length_list+input_length_list)
    # plt.title('Countplot for Output')
    # plt.xlabel('Values')
    # plt.ylabel('Count')
    # # ticks = [i for i in range(min(input_length_list),max(input_length_list),30)]
    # # labels = [str(i) for i in range(min(input_length_list),max(input_length_list),30)]
    # # plt.xticks(ticks=ticks,labels=labels)
    # plt.savefig('./countplot_for_total.png')
    # plt.show()
