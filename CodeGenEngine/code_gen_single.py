import os
import tvm
from tvm import (te,
                 auto_scheduler,
                 runtime,
                 topi)
import signal
from ops import TOPI_OPS_LIST,NN_OPS_LIST,CONV_OPS_LIST
from utils import (CodeGen,
                   ArgsGen,
                   write_ops_to_json,
                   write_ops_to_file,
                   handle_timeout
                   )
from args import args

print(args)

OP_TYPE_DICT = {'topi':TOPI_OPS_LIST,'nn':NN_OPS_LIST,'conv':CONV_OPS_LIST}
ARGS_PARAM_DICT = {'topi':['N','C','H','W'],
                   'nn':['N','C','H','W'],
                   'conv':['N','H','W','CO','CI','KH','KW','strides','padding']}

def generate_single_code(ops,max_attempts=1):
    print(ops.__name__)
    code_gen = CodeGen(search_time=10,log_file_path='./generate/log',verbose_key=False)
    args_gen = ArgsGen(param_list=ARGS_PARAM_DICT[args.op_type])
    attempts = 0
    while attempts < max_attempts:
        args_gen.randomize_params()
        op_args = args_gen.get_param_values()
        print(*op_args)
        signal.signal(signal.SIGALRM, handle_timeout)
        try:
            signal.alarm(args.timeout)
            c_code, ir_code = code_gen.c_codegen(topi_ops=ops, op_args=op_args)
            cuda_code = code_gen.cuda_codegen(topi_ops=ops, op_args=op_args)
            op_name = ops.__name__
            write_ops_to_json(op_name, c_code, cuda_code, ir_code,op_args)
            write_ops_to_file(op_name, c_code, cuda_code, ir_code,op_args)
            attempts += 1
        except TimeoutError:
            pass
        finally:
            signal.alarm(0)
        

def generate_code():
    for ops in OP_TYPE_DICT[args.op_type]:
        if ops.__name__ == args.op_name:
            generate_single_code(ops)


if __name__ == '__main__':
    generate_code()