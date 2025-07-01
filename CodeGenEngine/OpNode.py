import tvm
from tvm import topi
import json

file_path = '/code/OpGen-Verify/code_generation/ops/op_info.json'

class OpNode:
    def __init__(self,op_name,node_id=None) -> None:
        self.op_name = op_name
        self.node_id = f'{op_name}_{node_id}'
        self.extra_params = []
        self.input_tensors = []
        self.output_tensors = []
        """Load info from json file"""
        with open(file_path,'r') as file:
            op_info_list = json.load(file)
        self.load_info_from_json(op_info_list=op_info_list)
        """Init func call"""
        self.func_call = getattr(eval(self.func_call_str),self.op_name)

    
    def load_info_from_json(self,op_info_list):
        """Select op_info by name"""
        for _ in op_info_list:
            if _['op_name'] == self.op_name:
                op_info = _
        if op_info is None:
            raise ValueError(f'{self.op_name} is not found!')
        """Extract op info"""
        # input
        for key in op_info['input']:
            _input_info = op_info['input'][key]
            if _input_info['type'] == 'tensor':
                _input_info['dim'] = len(_input_info['shape'])
                self.input_tensors.append(_input_info)
            else:
                self.extra_params.append(_input_info)
        # output
        for key in op_info['output']:
            _output_info = op_info['output'][key]
            if _output_info['type'] == 'tensor':
                _output_info['dim'] = len(_output_info['shape'])
                self.output_tensors.append(_output_info)
            else:
                raise ValueError('Output type must be tensor!')
        self.func_call_str = op_info['func_call']

    def verify_input_tensor(self,verified_tensor_list,input_tensor):
        variable_dict = {}
        if not verified_tensor_list:
            return True
        """1. Both input tensors should have the same shape"""
        for verified_tensor in verified_tensor_list:
            for idx,_shape in enumerate(verified_tensor['shape']):
                if _shape not in variable_dict:
                    variable_dict[_shape] = verified_tensor['value'].shape[idx]
                else:
                    assert variable_dict[_shape] == verified_tensor['value'].shape[idx]
        return True
    
    def verify_output_tensor(self,output_tensor):
        if len(output_tensor.shape) == self.output_tensors[0]['dim']:
            return True
        else:
            return False        

    def set_input(self,input_tensor):
        verified_tensor_list = [] # the validated input tensors
        for idx,_tensor in enumerate(self.input_tensors):
            if 'value' in _tensor:
                verified_tensor_list.append(_tensor)
            else:
                # TODO: verify current input tensor
                is_legal = self.verify_input_tensor(verified_tensor_list,input_tensor)
                if is_legal:
                    self.input_tensors[idx]['value'] = input_tensor

    def set_output(self,output_tensor):
        if len(self.output_tensors) != 1:
            raise ValueError(f'Excepted output tensor length is 1 but get {len(self.output_tensors)}')
        is_legal = self.verify_output_tensor(output_tensor)
        if is_legal:
            self.output_tensors[0]['value'] = output_tensor

    def __str__(self) -> str:
        return f'{self.node_id}-{self.input_tensors}-{self.output_tensors}'
    

if __name__ == '__main__':
    node = OpNode('add')
    print(node.input_tensors,node.output_tensors)
    print(node.extra_params)
    print(node.func_call.__name__)