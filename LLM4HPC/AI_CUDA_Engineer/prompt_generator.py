import operator
from os.path import join, exists
import os
import re
import json
import ast


class PromptGenerator:
    TVM_PREFIX_PATTERN = re.compile(r"^(.*?)(_\[|_None)+?")
    TVM_INPUT_SHAPE_PATTERN = re.compile(
        r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])(?![^\[]*\])"
    )
    TVM_FUNCTION_PROTOTYPE_PATTERN = re.compile(R"^(.*?)\s*{")

    def __init__(self):
        self.template_base_path = (
            "/code/LLM4HPCTransCompile/LLM4HPC/AI_CUDA_Engineer/prompts"
        )
        self.kernel_bench_base_path = (
            "/code/LLM4HPCTransCompile/LLM4HPC/AI_CUDA_Engineer/KernelBench"
        )
        self.pytorch_code_functional_base_path = "/code/LLM4HPCTransCompile/LLM4HPC/AI_CUDA_Engineer/PyTorch_Code_Functional"
        # self.cuda_code_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/CUDA_Code'
        # self.cuda_code_base_path = '/code/LLM4HPCTransCompile/LLM4HPC/benchmark/cuda'
        self.cuda_code_base_path = "/code/LLM4HPCTransCompile/ParallelBench"
        self.cuda_code_base_tvm_path = (
            R"/code/LLM4HPCTransCompile/HPCTransEval/benchmark/data"
        )
        # self.c_code_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/CPU_Code'
        self.c_code_base_path = (
            "/code/LLM4HPCTransCompile/LLM4HPC/results/DeepSeekCoder_Lite"
        )
        self.generated_c_code_base_path = "/code/LLM4HPCTransCompile/Results"
        self.template_dict = {
            "conversion": {
                "BASE": "conversion_base_prompt_simplify.txt",
                "SYSTEM": "conversion_system_prompt.txt",
            },
            "optimization": {
                "BASE": "optimization_base_prompt.txt",
                "SYSTEM": "optimization_system_prompt.txt",
            },
            "translation": {
                "BASE": "translation_base_prompt.txt",
                "SYSTEM": "translation_system_prompt.txt",
            },
            "translation_c": {
                "BASE": "translation_c_base_prompt.txt",
                "SYSTEM": "translation_c_system_prompt.txt",
            },
            "translation_c_tvm": {
                "BASE": "translation_c_tvm_base_prompt.txt",
                "SYSTEM": "translation_c_tvm_system_prompt.txt",
            },
            "optimization_c": {
                "BASE": "optimization_c_correctness_base_prompt.txt",
                "SYSTEM": "optimization_c_correctness_system_prompt.txt",
            },
        }
        self.save_path_dict = {
            "conversion": "/code/LLM4HPCTransCompile/LLM4HPC/AI_CUDA_Engineer/PyTorch_Code_Functional",
            "translation": "/code/LLM4HPCTransCompile/LLM4HPC/AI_CUDA_Engineer/CUDA_Code",
            "translation_c": "/code/LLM4HPCTransCompile/Results",
            "translation_c_tvm": "/code/LLM4HPCTransCompile/Results",
            "optimization_c": "/code/LLM4HPCTransCompile/Results",
        }
        self.suffix_dict = {
            "conversion": ".py",
            "translation": ".cu",
            "translation_c": ".cpp",
            "translation_c_tvm": ".cpp",
            "optimization_c": ".cpp",
        }

        # Load tvm operator infos
        self.TVM_JSON_DICTS = {}
        self.TVM_JSON_FILE_PATHS = {}
        with open(
            R"/code/LLM4HPCTransCompile/HPCTransEval/benchmark/data/l1.json"
        ) as file:
            self.TVM_JSON_DICTS["level1"] = json.load(file)
            self.TVM_JSON_FILE_PATHS["level1"] = file.name
        with open(
            R"/code/LLM4HPCTransCompile/HPCTransEval/benchmark/data/l2.json"
        ) as file:
            self.TVM_JSON_DICTS["level2"] = json.load(file)
            self.TVM_JSON_FILE_PATHS["level2"] = file.name
        with open(
            R"/code/LLM4HPCTransCompile/HPCTransEval/benchmark/data/l3.json"
        ) as file:
            self.TVM_JSON_DICTS["level3"] = json.load(file)
            self.TVM_JSON_FILE_PATHS["level3"] = file.name

    def load_template(self, action="conversion"):
        # try:
        with open(
            join(self.template_base_path, self.template_dict[action]["BASE"])
        ) as file:
            base_template = file.read()
        with open(
            join(self.template_base_path, self.template_dict[action]["SYSTEM"])
        ) as file:
            system_template = file.read()
        # except FileNotFoundError:
        #     return 'File Not Found.'
        #     raise f'Error: {str(e)}'
        return base_template, system_template

    def load_source_code_single(
        self,
        level,
        operator,
        action="conversion",
        return_code=False,
        model_name=None,
    ):
        if action == "conversion":
            file_path = join(
                self.kernel_bench_base_path, level, f"{operator}.py"
            )
        elif action == "translation":
            file_path = join(
                self.pytorch_code_functional_base_path, level, f"{operator}.py"
            )
        elif action == "translation_c":
            file_path = join(self.cuda_code_base_path, level, f"{operator}.cu")
        elif action == "eval_c":
            file_path = join(self.c_code_base_path, level, f"{operator}.cpp")
        elif action == "optimization_c":
            file_path = join(
                self.generated_c_code_base_path,
                model_name,
                level,
                f"{operator}.cpp",
            )
        elif action == "translation_c_tvm":
            # Match {{ op_name }}_xxx_[xxx]
            if not (
                matched_prefix := PromptGenerator.TVM_PREFIX_PATTERN.findall(
                    operator
                )
            ):
                raise RuntimeError(
                    f"Op name not found in operator: {operator}"
                )
            op_name = matched_prefix[0][0]

            # Match xxx_xxx_[xxx]_{{ op_input }}
            if not (
                matched_input := PromptGenerator.TVM_INPUT_SHAPE_PATTERN.findall(
                    operator
                )
            ):
                raise RuntimeError(
                    f"Op input not found in operator: {operator}"
                )
            op_input_shape = matched_input[-1]

            # Find op output shape in json
            if not (
                target_op_dict := PromptGenerator.find_op_by_name_and_shape(
                    self.TVM_JSON_DICTS[level], op_name, op_input_shape
                )
            ):
                raise RuntimeError(
                    f"Op output shape not found in file "
                    f"{self.TVM_JSON_FILE_PATHS[level]}, given op name: "
                    f"{op_name} and input shape: {op_input_shape}"
                )

            op_output_shape = target_op_dict["output_shape"]
            op_function_prototype = (
                PromptGenerator.TVM_FUNCTION_PROTOTYPE_PATTERN.findall(
                    target_op_dict["c_code"]
                )[0]
            )
            file_path = join(
                self.cuda_code_base_tvm_path, level, f"{operator}.cu"
            )

        if not return_code:
            raise NotImplementedError("Returning file path is not supportted.")

        with open(file_path, "r", encoding="utf-8") as file:
            code = file.read()

        if action == "translation_c_tvm":
            # Return code, op name, op input, op_output_shape
            return (
                code,
                op_name,
                op_input_shape,
                op_output_shape,
                op_function_prototype,
            )

        return code

    def fill_template(self, source_code, action="conversion", operator=None):
        base_template, system_template = self.load_template(action)
        if action == "conversion":
            base_content = base_template.replace("{code}", source_code)
            system_content = system_template
        elif action == "translation":
            base_content = base_template.replace("{EXAMPLES}", source_code)
            system_content = system_template
        elif action == "translation_c":
            base_content = base_template.replace("{EXAMPLES}", source_code)
            system_content = system_template
        elif action == "translation_c_tvm":
            (
                code,
                op_name,
                op_input_shape,
                op_output_shape,
                op_func_prototype,
            ) = source_code
            base_content = base_template.replace(R"{EXAMPLES}", code)
            system_content = (
                system_template.replace(R"{{ OP_NAME }}", op_name)
                .replace(R"{{ INPUT_SHAPE }}", str(op_input_shape))
                .replace(R"{{ OUTPUT_SHAPE }}", str(op_output_shape))
                .replace(R"{{ FUNC_PROTOTYPE }}", op_func_prototype)
            )
        elif action == "optimization_c":
            base_content = base_template.replace("{EXAMPLES}", source_code)
            system_content = system_template
            # system_content = system_template.replace("{operation}",operator)
        return base_content, system_content

    def load_source_code(self, levels=["level1"]):
        """
        level: ['level1','level2','level3','level4']
        """
        source_code_dict = {}
        for level in levels:
            source_code_dict[level] = []
            level_folder_path = join(self.kernel_bench_base_path, level)
            operator_list = os.listdir(level_folder_path)
            for operator in operator_list:
                python_code = self.load_source_code_single(level, operator)
                source_code_dict[level].append(python_code)
        return source_code_dict

    def generate_prompt(self):
        pass

    @staticmethod
    def find_op_by_name_and_shape(
        json_data, target_op_name, target_input_shape
    ):
        if isinstance(target_input_shape, str):
            target_input_shape = ast.literal_eval(target_input_shape)

        # Iterator through all op dicts
        for op in json_data:
            if op.get("op_name") == target_op_name:
                current_input_shape = op.get("input_shape")

                if current_input_shape == target_input_shape:
                    return op

        return None


if __name__ == "__main__":
    prompt_generator = PromptGenerator()
    base_template, system_template = prompt_generator.load_template()
    python_code = prompt_generator.load_source_code_single(
        "level1", "1_Square_matrix_multiplication_.py"
    )
    base_content, system_content = prompt_generator.fill_template(python_code)
    print(base_content)
    print("@@@@@@@@@@@@@@@@")
    print(system_content)
