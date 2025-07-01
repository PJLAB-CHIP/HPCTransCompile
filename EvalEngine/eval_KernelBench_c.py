from calendar import c
from dataclasses import dataclass
import shutil
import time
import pydra
# from pydra import REQUIRED, Config

import json
import argparse
from tqdm import tqdm
# from src import eval, utils
import torch
import torch.nn as nn
import os
import multiprocessing as mp
import numpy as np
from torch.utils.cpp_extension import load

# from datasets import load_dataset
# from src.eval import register_and_format_exception, KernelExecResult, check_metadata_serializable_all_types
# from src.utils import read_file
from pydantic import BaseModel


"""
Batch Evaluation from Existing Generated CPU Codes

This expects you have generated the kernels and stored them in the runs/{run_name} directory
This eval script will evaluate the kernels against the reference architecture, and store the results in the runs/{run_name}/eval_results.json file

Usually with eval, we check
- correctness (n_correct): 5 randomized input trials
- performance (n_trials): 100 randomized input trials

You can increase the number of trials for correctness and performance
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class EvalConfig():
    def __init__(self):

        self.run_name = "TestDev" # name of the run to evaluate, aka, model generated results path
        # self.run_name = "QwenCoder_14b" #

        self.dataset_src = "local"

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = 1

        # subset of problems to evaluate
        self.subset = (None, None) # (start_id, end_id), these are the logical index

        # Evaluation Mode: local (requires GPU), see modal (cloud GPU) in the modal file
        self.eval_mode = "local"

        # # Construct this from mapping from architecture name to torch cuda arch list in the future
        # # you can either specify SM version or just use the name
        # self.gpu_arch = ["Ada"]

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "Results")

        self.verbose = False

        # Eval settings
        self.num_correct_trials = 5
        self.num_perf_trials = 30
        self.timeout = 180 # in seconds
        self.measure_performance = True
        self.compile_timeout= 60
        
        # Eval Flow setting
        # To speedup evaluation, you can start building the kernel on CPU on disk as cache
        self.build_cache = False
        self.num_cpu_workers = 20 # number of parallel process to to parallelize the build on CPUs
        
        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join(self.runs_dir, "cache")

        # number of CPUs to do batch evaluation, cpu kernel may use multi-cpu cores to speed up?
        self.num_cpu_devices = 1
        

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int
    sample_id: int
    device: torch.device


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance
    torch_runtime : float = -1.0
    torch_runtime_stats : dict = {}

def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def fetch_ref_arch_from_problem_id(dataset, problem_id: int, dataset_src: str) -> str | None:
    """
    Fetch reference architecture from problem directory
    Either from Hugging Face or Local Dataset
    """
    ref_arch_path = dataset[problem_id]
    problem_name = os.path.basename(ref_arch_path)
    ref_arch_src = read_file(ref_arch_path)

    # verify
        # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    
    return ref_arch_src

def parse_cpu_kernel_id(file_name):
        file_id = file_name.split("_")[0]
        return int(file_id)

def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> str | None:
    """
    Fetch kernel file from disk, to load cpu kernel test case, just ret the kernel file full path
    """
    level_path = os.path.join(run_dir, "level" + str(level))

    # get cpu kernel string
    kernel_files = [f for f in os.listdir(level_path) \
                                if f.endswith(".cpp") and \
                                    (problem_id == parse_cpu_kernel_id(f))]
    if not kernel_files:
        raise RuntimeError("No CPU kernel file found")
    
    kernel_file = os.path.join(level_path, kernel_files[0])
    return kernel_file

def set_seed(seed: int):
    torch.manual_seed(seed)

def cpu_graceful_eval_cleanup(curr_context: dict, device: torch.device):
    """
    Clean up env after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context

def run_and_check_correctness_cpu(
    nn_model_instance: nn.Module,
    module_fn: callable, # type: ignore
    custom_fn: callable, # type: ignore
    get_inputs_fn: callable, # type: ignore
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed) # type: ignore
            inputs = get_inputs_fn()
            inputs = [
                x.cpu() if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed) # type: ignore
            model = nn_model_instance.cpu()
            output = model(*inputs, fn=module_fn)
            # print("original method output: ", output, output.shape)

            try:
                output_new = model(*inputs, fn=custom_fn)
                # print("custom method output: ", output_new, output_new.shape)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for custom cpu kernel: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
    

def time_execution_with_cpu(
    nn_module_instance: nn.Module,
    kernel_fn: callable, # type: ignore
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None, # type: ignore
) -> list[float]:
    """
    Time a CPU kernel function over multiple trials 
    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CPU device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cpu.current_device()}")
        device = torch.cpu.current_device()
    
    # Warm ups
    with torch.no_grad():
        for _ in range(num_warmup):
            nn_module_instance(*args, fn=kernel_fn)

    print(
        f"[Profiling] Using device: {device} , warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    with torch.no_grad():
        # Actual trials
        for trial in range(num_trials):
            # create event marker default is not interprocess
            start_time = time.perf_counter()
            nn_module_instance(*args, fn=kernel_fn)
            end_time = time.perf_counter()

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = (end_time - start_time) * 1000
            if verbose:
                print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def get_timing_stats_cpu(elapsed_times: list[float], device: torch.device = None) -> dict: # type: ignore
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: cpu device
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = "cpu"
        stats["device"] = str(device)  # for debugging

    return stats

def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable, callable]: # type: ignore
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None # type: ignore

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None # type: ignore

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    module_fn = context.get("module_fn")
    return (Model, get_init_inputs_fn, get_inputs_fn, module_fn) # type: ignore


def load_custom_module(custom_kernel_file):
    task_name = custom_kernel_file.split("/")[-1].split(".")[0]
    pid_str = task_name.split("_")[0]
    task_name = "_".join(task_name.split("_")[1:])   # Remove problem ID
    task_name = task_name + "_" + pid_str
    if task_name == "":
        task_name = "task"
    print(task_name)

    cpu_module = load(
        name=task_name,
        sources=[custom_kernel_file],
        extra_cflags=[
        '-O3',
        '-fopenmp',         # 启用 OpenMP
        '-mavx2',           # 启用 AVX2
        ],
        extra_ldflags=['-lgomp'],
        verbose=True,
    )
    return cpu_module

def eval_kernel_against_ref_cpu(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None, # type: ignore
    device: torch.device = torch.device("cpu"), # have to run on cpu
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    """
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    context = {}
    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs, module_fn = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cpu() if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CPU Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = "cpu"
    metadata["device"] = str(device)  # for debugging

    # this is where compilation happens
    try:
        custom_module = load_custom_module(custom_model_src)
        assert hasattr(custom_module, "forward")
    except Exception as e:
        print(
            f"Failed to compile custom CPU kernel: Record as compilation failure. \nError: {e}"
        )

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            cpu_graceful_eval_cleanup(context, device)
            return None # type: ignore
        else:
            metadata["compilation_error"] = e
            cpu_graceful_eval_cleanup(context, device)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps
    
    kernel_exec_result = None
    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness_cpu(
            original_model,
            module_fn,
            custom_module.forward, # type: ignore
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                with torch.no_grad():
                    set_seed(seed_num)
                    inputs = get_inputs()
                    inputs = [
                        x.cpu() if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ]
                    elapsed_times = time_execution_with_cpu(
                        original_model,
                        module_fn, # type: ignore
                        *inputs,
                        num_trials=num_perf_trials,
                        verbose=verbose,
                        device=device,
                    )
                    torch_runtime_stats = get_timing_stats_cpu(elapsed_times, device=device)

                with torch.no_grad():
                    set_seed(seed_num)
                    inputs = get_inputs()
                    inputs = [
                        x.cpu() if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ]
                    elapsed_times = time_execution_with_cpu(
                        original_model,
                        custom_module.forward, # type: ignore
                        *inputs,
                        num_trials=num_perf_trials,
                        verbose=verbose,
                        device=device,
                    )
                    runtime_stats = get_timing_stats_cpu(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                    print(f"[Eval] Performance Stats Torch: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
                kernel_exec_result.torch_runtime = torch_runtime_stats["mean"]
                kernel_exec_result.torch_runtime_stats = torch_runtime_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    cpu_graceful_eval_cleanup(context, device)
    return kernel_exec_result

def compile_single_sample(work_args: WorkArgs, configs: EvalConfig, dataset, run_dir: str):
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # fetch kernel from disk
    kernel_src = fetch_kernel_from_disk(run_dir, configs.level, problem_id, sample_id) # type: ignore
    assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

    custom_module = load_custom_module(kernel_src)
    assert hasattr(custom_module, "forward")
    return None

def evaluate_single_sample(work_args: WorkArgs, configs: EvalConfig, dataset, run_dir: str) -> KernelExecResult | None:
    """
    Evaluate a single sample on CPU
    """
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # fetch reference architecture from problem directory
    ref_arch_src = fetch_ref_arch_from_problem_id(dataset, problem_id, configs.dataset_src) # type: ignore

    # fetch kernel from disk
    # Add database support in the future
    kernel_src = fetch_kernel_from_disk(run_dir, configs.level, problem_id, sample_id) # type: ignore

    assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

    build_dir = os.path.join(configs.kernel_eval_build_dir, configs.run_name, f"{problem_id}", f"{sample_id}") # type: ignore

    try: 
        eval_result = eval_kernel_against_ref_cpu(
            original_model_src=ref_arch_src, # type: ignore
            custom_model_src=kernel_src,
            measure_performance=configs.measure_performance,
            verbose=configs.verbose,    
            num_correct_trials=configs.num_correct_trials,
            num_perf_trials=configs.num_perf_trials,
            build_dir=build_dir,
            device=device,
        )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} "
        )
        metadata = {"other_error": f"error: {str(e)}",
                    "hardware": "cpu",
                    "device": str(device)
                    } # for debugging
        eval_result = KernelExecResult(compiled=False, correctness=False, 
                                            metadata=metadata)
        return eval_result

def remove_cache_dir(cache_dir: str, run_name: str, problem_id, sample_id):
    """
    Remove the cached folder for sample compilation so it can start a clean build next time
    useful for time out, failed build, etc.
    """
    problem_cache_dir = os.path.join(cache_dir, run_name, f"{problem_id}", f"{sample_id}")
    print(f"cache_dir to remove: {problem_cache_dir}")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"\n[INFO] Removed cached folder for Problem ID: {problem_id}, Sample ID: {sample_id}")
        except Exception as e:
            print(f"\n[WARNING] Failed to remove cache directory {cache_dir}: {str(e)}")

def batch_compile(
    total_work: list[tuple[int, int]],
    config: EvalConfig,
    curr_level_dataset,
    run_dir: str,
    compile_file_path: str,
):
    """
    Batch evaluation across CPUs, do batch_size of work one on each cpu all at once
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    """
    # construct a list of work args
    batch_size = 20

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {batch_size} CPUs; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size, f"Current batch size {len(curr_work_batch)} is greater than the number of CPUs {batch_size}"

            with mp.Pool(batch_size) as pool:
                work_args = [(WorkArgs(problem_id=p_id, sample_id=s_idx, device=torch.device("cpu"),), config, curr_level_dataset, run_dir,)
                                for i, (p_id, s_idx) in enumerate(curr_work_batch)]
                start_time = time.time()
                async_results = []
                for work_arg in work_args:
                    async_results.append(pool.apply_async(compile_single_sample, work_arg))

                # Collect results with a batch timeout
                compile_results = []
                batch_compile_timeout = config.compile_timeout
    
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id = curr_work_batch[i]

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_compile_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        compile_results.append((problem_id, sample_id, "pass"))
                    except mp.TimeoutError:
                        print(f"[WARNING] Compilation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}")

                        compile_results.append((problem_id, sample_id, "Compilation Timeout"))
                        remove_cache_dir(config.kernel_eval_build_dir, config.run_name, problem_id, sample_id) # type: ignore
                    except Exception as e:
                        print(f"[ERROR] Compilation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}")

                        compile_results.append((problem_id, sample_id, str(e)))
                        remove_cache_dir(config.kernel_eval_build_dir, config.run_name, problem_id, sample_id) # type: ignore
        
                end_time = time.time()
                for problem_id, sample_id, result in compile_results:
                    print("-" * 128)
                    print(f"[Compilation Result] Problem ID: {problem_id}, Sample ID: {sample_id}")
                    print(result)
                    
                    if result == "pass":
                         kernel_result = KernelExecResult(compiled=True, correctness=False)
                         add_to_eval_results_file(problem_id, sample_id, kernel_result, compile_file_path)
                    else:
                        metadata = {"other_error": f"error: {result}", "hardware": "cpu", "device": "cpu"} # for debugging
                        kernel_result = KernelExecResult(compiled=False, correctness=False, metadata=metadata)
                        add_to_eval_results_file(problem_id, sample_id, kernel_result, compile_file_path)
                print("-" * 128)
                print(f"[Curr batch] Compilation took {end_time - start_time:.2f} seconds")
                pbar.update(len(curr_work_batch))


def batch_eval(
    total_work: list[tuple[int, int]],
    config: EvalConfig,
    curr_level_dataset,
    run_dir: str,
    eval_file_path: str,
):
    """
    Batch evaluation across CPUs, do batch_size of work one on each cpu all at once
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    """
    # construct a list of work args
    batch_size = config.num_cpu_devices
    batch_size = 1

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {config.num_cpu_devices} CPUs; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size, f"Current batch size {len(curr_work_batch)} is greater than the number of CPUs {batch_size}"

            ##########################-------------- Original load & correctness check & perf measurement ------------------##########################
            # will load cached compiled .so
            with mp.Pool(batch_size) as pool:
                work_args = [
                    (
                        WorkArgs(
                            problem_id=p_id,
                            sample_id=s_idx,
                            device=torch.device("cpu"),
                        ),
                        config,
                        curr_level_dataset,
                        run_dir,
                    )
                    for i, (p_id, s_idx) in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(evaluate_single_sample, work_arg)
                    )
            
                # Collect results with a batch timeout
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id = curr_work_batch[i]

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((problem_id, sample_id, result))
                        
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        metadata = {"other_error": f"error: Evaluation TIMED OUT", "hardware": "cpu", "device": "cpu"} # for debugging
                        fail_result = KernelExecResult(compiled=True, correctness=False, metadata=metadata)
                        results.append((problem_id, sample_id, fail_result))
                        remove_cache_dir(config.kernel_eval_build_dir, config.run_name, problem_id, sample_id) # type: ignore
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        metadata = {"other_error": f"error: {str(e)}", "hardware": "cpu", "device": "cpu"} # for debugging
                        fail_result = KernelExecResult(compiled=True, correctness=False, metadata=metadata)
                        results.append((problem_id, sample_id, fail_result))
                        remove_cache_dir(config.kernel_eval_build_dir, config.run_name, problem_id, sample_id) # type: ignore
                end_time = time.time()

            # current batch summary
            for problem_id, sample_id, result in results:
                print("-" * 128)
                print(
                    f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}"
                )
                print(result)

                # add all the batch results here to avoid file race condition
                # add to eval result if valid result
                if result is not None:
                    print(f"Adding Eval Result to file for problem {problem_id} sample {sample_id}")
                    add_to_eval_results_file(problem_id, sample_id, result, eval_file_path)

            print("-" * 128)
            print(
                f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
            )
            pbar.update(len(curr_work_batch))

def check_if_eval_exists_local(problem_id: int, sample_id: int, eval_file_path: str) -> bool:
    """
    Check if evaluation result already exists in eval results file
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
        return str(problem_id) in eval_results
    return False

def check_compile_pass(problem_id: int, sample_id: int, compile_file_path: str) -> bool:
    if os.path.exists(compile_file_path):
        with open(compile_file_path, 'r') as f:
            compile_res = json.load(f)        
        return compile_res[str(problem_id)]["compiled"]
    return True

def add_to_eval_results_file(problem_id: int, sample_id: int, eval_result: KernelExecResult, eval_file_path: str):
    """
    Add evaluation result to eval results file
    TODO: migrate database support
    """
    # Load existing results if file exists
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
    
    # Add new result
    eval_results[str(problem_id)] = {
        # assume 1 sample for now, will think about how to do this better for more samples
        'sample_id': sample_id,
        'compiled': eval_result.compiled,
        'correctness': eval_result.correctness,
        'metadata': check_metadata_serializable_all_types(eval_result.metadata),
        'runtime': eval_result.runtime,
        'runtime_stats': eval_result.runtime_stats,
        'torch_runtime' : eval_result.torch_runtime,
        'torch_runtime_stats' : eval_result.torch_runtime_stats
    }
    
    # Write updated results back to file
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f)

KERNEL_BENCH_PATH = "/code/LLM4HPCTransCompile/EvalEngine/torch_functionals"
def construct_problem_dataset_from_problem_dir(problem_dir: str):
    """
    Construct a list of relative paths to all the python files in the problem directory
    Sorted by the numerical prefix of the filenames
    """
    DATASET = []

    for file_name in os.listdir(problem_dir):
        if file_name.endswith(".py"):
            # TODO: revisit later to satisfy eval harnes
            relative_path = os.path.join(problem_dir, file_name)
            DATASET.append(relative_path)

    # Sort the DATASET based on the numerical prefix of the filenames
    DATASET.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))

    DATASET_DICT = {}   # problem_id -> file_path
    for file_path in DATASET:
        file_name = file_path.split("/")[-1]
        problem_id = int(file_name.split("_")[0])
        DATASET_DICT[problem_id] = file_path
    return DATASET_DICT


def construct_kernelbench_dataset(level: int):
    return construct_problem_dataset_from_problem_dir(
        os.path.join(KERNEL_BENCH_PATH, f"level{level}")
    )


def fetch_valid_cpp_ids(run_dir, level):
    level_path = os.path.join(run_dir, "level" + str(level))

    # get cpu kernel string
    kernel_files = [f for f in os.listdir(level_path) if f.endswith(".cpp")]
    valid_kernel_ids = [parse_cpu_kernel_id(f) for f in kernel_files]
    return valid_kernel_ids

def add_compile_fail_in_eval(compile_file_path, eval_file_path):
    if os.path.exists(compile_file_path) and os.path.exists(eval_file_path):
        compile_fail_res ={}
        with open(compile_file_path, 'r') as f:
            compile_res = json.load(f)
            for pid, res in compile_res.items():
                if not res["compiled"]:
                    problem_id = int(pid)
                    kernel_res = KernelExecResult(compiled=False, correctness=False, metadata=res["metadata"])
                    add_to_eval_results_file(problem_id, 0, kernel_res, eval_file_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate transpiled CPU code.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the run (e.g. TestDev)')
    parser.add_argument('--level', type=int, required=True,
                        help='Level number (e.g. 1)')
    return parser.parse_args()

# @pydra.main(base=EvalConfig)
def main():
    """
    Batch Eval Samples from Particular Run
    Store Eval Results in specified eval results file
    """

    args = parse_args()
    config = EvalConfig()
    config.run_name = args.run_name
    config.level = args.level

    print(f"Starting Batch Eval with config: {config.run_name}, {config.level}")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    curr_level_dataset = construct_kernelbench_dataset(config.level) # type: ignore
    num_problems_in_level = len(curr_level_dataset)
    curr_level_problem_ids = [idx for idx in curr_level_dataset.keys()]

    if config.subset == (None, None):
        problem_id_range = curr_level_problem_ids
    else:
        subset_ids = list(range(config.subset[0], config.subset[1])) # type: ignore
        interset_ids = list(set(curr_level_problem_ids) & set(subset_ids))
        problem_id_range = interset_ids # type: ignore

    run_dir = os.path.join(config.runs_dir, config.run_name) # type: ignore
    eval_file_path = os.path.join(run_dir, f"{config.level}_eval_results.json")
    compile_file_path = os.path.join(run_dir, f"{config.level}_compile_result.json")

    valid_cpp_ids = fetch_valid_cpp_ids(run_dir, config.level)
    problem_id_range = sorted(list(set(problem_id_range) & set(valid_cpp_ids)))
    assert len(problem_id_range) > 0

    
    total_work = []
    for problem_id in problem_id_range:
        sample_id = 0 # only evaluate 1 sample for now
        if not check_if_eval_exists_local(problem_id, sample_id, compile_file_path):
            total_work.append((problem_id, sample_id))
    print(f"Start compilation on {len(total_work)} unevaluated samples in range: {problem_id_range}")
    # Batch Compile all modules first
    batch_compile(total_work, config, curr_level_dataset, run_dir, compile_file_path)

    print(num_problems_in_level, run_dir, eval_file_path)
    print(f"Evaluating 1 sample each for level {config.level} problems: {problem_id_range}")
    total_work = []
    for problem_id in problem_id_range:
        sample_id = 0 # only evaluate 1 sample for now
        if not check_if_eval_exists_local(problem_id, sample_id, eval_file_path) and \
               check_compile_pass(problem_id, sample_id, compile_file_path):
            total_work.append((problem_id, sample_id))
    print(f"Start evaluation on {len(total_work)} unevaluated samples in range: {problem_id_range}")

    # Batch Eval on CPUs
    batch_eval(total_work, config, curr_level_dataset, run_dir, eval_file_path)
    # add compilation fail results in eval_file
    add_compile_fail_in_eval(compile_file_path, eval_file_path)


if __name__ == "__main__":
    main()