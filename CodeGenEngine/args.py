import argparse


"""命令行解析参数信息"""
parser = argparse.ArgumentParser()
parser.add_argument('--op_type',type=str,default='topi',
                    help='the op type to generate')
parser.add_argument('--op_name',type=str,default='abs',
                    help='the op type to generate')
parser.add_argument('--timeout',type=int,default=60,
                    help='time to wait for a single op generation')
args = parser.parse_args()  # 解析命令行参数