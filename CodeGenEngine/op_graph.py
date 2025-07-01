from collections import defaultdict
import random
from re import L
from tvm import (
    te,
    auto_scheduler,
    topi
)

DEFAULT_OP = 'ph'

OP_DICT = {
    'add': 2,
    'subtract': 2,
    'matmul':2,
    'batch_matmul':2,
    'dense':2,
    'cos': 1,
    'cosh': 1,
    'sin': 1,
    'sinh': 1,
    'abs': 1,
    'tan': 1,
    'tanh': 1
}

OP_TE_DICT = {
    'add': topi.add,
    'subtract': topi.subtract,
    'matmul':topi.nn.matmul,
    'batch_matmul':topi.nn.batch_matmul,
    'dense':topi.nn.dense,
    'cos': topi.cos,
    'cosh': topi.cosh,
    'sin': topi.sin,
    'sinh': topi.sinh,
    'abs': topi.abs,
    'tan': topi.tan,
    'tanh': topi.tanh
}

class OpNode:
    def __init__(self,op_name,node_id=None,input_tensors=None,output_tensors=None) -> None:
        self.op_name = op_name
        self.node_id = f'{op_name}_{node_id}'
        self.input_tensors = []
        self.output_tensors = []
    def set_input(self,input_tensors):
        self.input_tensors.append(input_tensors)

    def set_output(self,output_tensors):
        self.output_tensors.append(output_tensors)

    def __str__(self) -> str:
        return f'{self.node_id}-{self.input_tensors}-{self.output_tensors}'


class OpGraph:
    def __init__(self) -> None:
        self.counter = 0
        self.adjacency_list = defaultdict(list)
        self.in_degree = {}
        self.leaf_list = [] # (output)
        self.node_list = [] # all operators
        self.input_list = [] # te.placehoder(input)
        """init OpGraph"""
        self.add_vertex('ph')

    def __str__(self) -> str:
        result = ""
        for node, neighbors in self.adjacency_list.items():
            neighbors_name = []
            for neighbor in neighbors:
                neighbors_name.append(neighbor.node_id)
            result += f"{node}: {neighbors_name}\n"
        return result
    
    def add_vertex(self, op_name):
        # init node info
        # (1) create OpNode instance
        node = OpNode(op_name,self.counter)
        self.counter += 1
        self.adjacency_list[node] = []
        self.in_degree[node] = 0
        self.node_list.append(node)
        self.leaf_list.append(node)
        if op_name == 'ph':
            self.input_list.append(node)
        return node

    def add_edge(self,source_node,target_node):
        # init edge info
        print('source_node,target_node:', source_node,target_node)
        self.adjacency_list[source_node].append(target_node)
        self.in_degree[target_node] += 1
        if source_node in self.leaf_list:
            self.leaf_list.remove(source_node)

        # Infer input and output tensor info
        # source_op_name = source_node.op_name
        # if source_op_name == 'ph':
        #     data = te.placeholder((2,19),name=f'{source_node.node_id}')
        #     source_node.set_input(data)
        #     source_node.set_output(data)
        #     target_node.set_input(data)
        # else:
        #     assert len(source_node.input_tensors) == OP_DICT[source_op_name]
        #     # execute the func
        #     if OP_DICT[source_op_name] == 1:
        #         out = OP_TE_DICT[source_op_name](source_node.input_tensors[0])
        #     elif OP_DICT[source_op_name] == 2:
        #         if source_op_name == 'matmul':
        #             out = OP_TE_DICT[source_op_name](source_node.input_tensors[0],source_node.input_tensors[1],transp_b=True)
        #         else:
        #             out = OP_TE_DICT[source_op_name](source_node.input_tensors[0],source_node.input_tensors[1])
        #     source_node.set_output(out)
        #     target_node.set_input(out)

    def extend_graph(self, op_name):
        print('self.node_list:', self.node_list)
        print('self.leaf_list:', self.leaf_list)

        """1. add te.placehoder(input)"""
        if OP_DICT[op_name]>len(self.node_list):
            for _ in range(OP_DICT[op_name]-len(self.node_list)):
                self.add_vertex('ph')
        random.seed(42)

        # TODO: Check whether selected node can form legal computation step
        selected_nodes = random.sample(self.node_list,OP_DICT[op_name])
        new_node = self.add_vertex(op_name)
        print('selected_nodes:', selected_nodes)
        for _ in selected_nodes:
            self.add_edge(_,new_node)
    
    def select_op_shape(self,op_node,N,C,H,W,CO,CI,KH,KW,strides,padding):
        _shape = (N,C)
        """element-wise"""
        """matmul"""
        if op_node.op_name == 'matmul':
            if len(op_node.input_tensors) == 0:
                pass
            elif len(op_node.input_tensors) == 1:
                lshape = op_node.input_tensors[0].shape
                assert len(lshape)>=2
                lshape[-1], lshape[-2] = lshape[-2], lshape[-1]
                _shape = lshape
        """two input"""
        return _shape

    def topological_sort(self):
        queue = [vertex for vertex in self.in_degree if self.in_degree[vertex] == 0]
        topological_order = []

        while queue:
            vertex = queue.pop(0)
            topological_order.append(vertex)

            for neighbor in self.adjacency_list[vertex]:
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topological_order) != len(self.in_degree):
            return None  # 存在环,无法求出拓扑序列
        else:
            return topological_order

    def extract_op_name(self):
        topo_order = self.topological_sort()
        if topo_order:
            print("Topological order:", topo_order)
        else:
            print("Graph contains a cycle, no topological order exists.")
        op_name = ''
        for i,step in enumerate(topo_order):
            if i != 0:
                op_name += '-'
            op_name += step.node_id
        print('op_name:', op_name)
        return op_name

    def save_to_json(self,file_path='./generate/graph'):
        pass
def add_calculation_steps(op_graph:OpGraph):
    print('==================')
    op_graph.extend_graph('add')
    # print('==================')
    # op_graph.extend_graph('abs')
    # print('==================')
    # op_graph.extend_graph('abs')
    # print('==================')
    # op_graph.extend_graph('matmul')
    # print('==================')
    # op_graph.extend_graph('cos')
    # print('==================')
        
@auto_scheduler.register_workload
def topology_expansion(N,C,H,W,CO,CI,KH,KW,strides,padding):
    """1. Init op graph"""
    op_graph = OpGraph()

    """2. Add calculation steps"""
    add_calculation_steps(op_graph) # check if the step is legal at the same time

    """3. Get topology order"""
    topo_order = op_graph.topological_sort()
    op_graph.extract_op_name()
    if topo_order:
        print("Topological order:", topo_order)
    else:
        print("Graph contains a cycle, no topological order exists.")
    print('op_graph:', op_graph)
    print('op_graph.leaf_list:', op_graph.leaf_list)
    print('op_graph.input_list:', op_graph.input_list)

    """4. Execute op graph"""
    for step in topo_order:
        print('step.node_id:', step.node_id)
        _op_name = (step.node_id).split('_')[0]
        if _op_name == 'ph':
            _shape = op_graph.select_op_shape(step,N,C,H,W,CO,CI,KH,KW,strides,padding)
            print('_shape:', _shape)
            data = te.placeholder(_shape,name=f'{step.node_id}')
            step.set_input(data)
            step.set_output(data)
            for next_step in op_graph.adjacency_list[step]:
                next_step.set_input(data)
        else:
            assert len(step.input_tensors) == OP_DICT[_op_name]
            # execute the func
            if OP_DICT[_op_name] == 1:
                out = OP_TE_DICT[_op_name](step.input_tensors[0])
            elif OP_DICT[_op_name] == 2:
                if _op_name == 'matmul':
                    out = OP_TE_DICT[_op_name](step.input_tensors[0],step.input_tensors[1],transpose_b=True)
                else:
                    out = OP_TE_DICT[_op_name](step.input_tensors[0],step.input_tensors[1])
            step.set_output(out)
            for next_step in op_graph.adjacency_list[step]:
                next_step.set_input(out)
    for step in topo_order:
        print(step)

    """5. Extract input and output tensor"""
    ret = []
    for _node in op_graph.input_list:
        ret.extend(_node.input_tensors)
    for _node in op_graph.leaf_list:
        ret.extend(_node.output_tensors)
    return ret

if __name__ == '__main__':
    """Fixed Combination"""
    topology_expansion()

    """Flexible Combination"""