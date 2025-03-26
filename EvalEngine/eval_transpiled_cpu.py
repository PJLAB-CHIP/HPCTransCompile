from calendar import c
from dataclasses import dataclass
import shutil
import time
import pydra
from pydra import REQUIRED, Config

import json
from tqdm import tqdm
from src import eval, utils
import torch
import torch.nn as nn
import os
import multiprocessing as mp
import numpy as np
from torch.utils.cpp_extension import load

from datasets import load_dataset
from src.eval import register_and_format_exception, KernelExecResult, check_metadata_serializable_all_types
from src.utils import read_file


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


class EvalConfig(Config):
    def __init__(self):

        self.run_name = REQUIRED # name of the run to evaluate, aka, model generated results path
        # self.run_name = "QwenCoder_14b" #

        self.dataset_src = "local"

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

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
        self.num_perf_trials = 100
        self.timeout = 180 # in seconds
        self.measure_performance = True
        
        # Eval Flow setting
        # To speedup evaluation, you can start building the kernel on CPU on disk as cache
        self.build_cache = False
        self.num_cpu_workers = 20 # number of parallel process to to parallelize the build on CPUs
        
        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")

        # number of CPUs to do batch evaluation, cpu kernel may use multi-cpu cores to speed up?
        self.num_cpu_devices = 1
        

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int
    sample_id: int
    device: torch.device


def fetch_ref_arch_from_problem_id(dataset, problem_id: int, dataset_src: str) -> str | None:
    """
    Fetch reference architecture from problem directory
    Either from Hugging Face or Local Dataset
    """
    if dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    
    elif dataset_src == "local":
        problem_idx_in_dataset = problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # verify
        # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    
    return ref_arch_src


def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> str | None:
    """
    Fetch kernel file from disk, to load cpu kernel test case, just ret the kernel file full path
    """
    level_path = os.path.join(run_dir, "level" + str(level))

    def parse_cpu_kernel_id(file_name):
        file_id = file_name.split("_")[0]
        print("@@@@@@---> hit file_id: ", int(file_id))
        return int(file_id)

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
            print("original method output: ", output, output.shape)

            try:
                output_new = model(*inputs, fn=custom_fn)
                print("custom method output: ", output_new, output_new.shape)
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
    for _ in range(num_warmup):
        nn_module_instance(*args, fn=kernel_fn)

    print(
        f"[Profiling] Using device: {device} , warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

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
    task_name = "_".join(task_name.split("_")[1:])   # Remove problem ID
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

    print("?????????????enter the original model load", custom_model_src)

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

    print("@@@@@@-----> Finish correctness check\n")

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

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
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    cpu_graceful_eval_cleanup(context, device)
    print("@@@@@@-----> Finish all test check\n")
    return kernel_exec_result


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
    
# def cuda_single_eval_wrapper(curr_work: WorkArgs, configs: dict, dataset, run_dir: str):
#     """
#     Wrapper to handle timeout and keyboard interrupt
#     """

#     with mp.Pool(1) as pool:
#         try:
#             result = pool.apply_async(
#                 evaluate_single_sample,
#                 args=(curr_work, configs, dataset, run_dir),
#             ).get(timeout=configs.timeout)
#         except KeyboardInterrupt:
#             print(
#                 "\n [Terminate] Caught KeyboardInterrupt, terminating workers..."
#             )
#             pool.terminate()
#             pool.join()
#             raise
#         except mp.TimeoutError as e:
#             print(f"[WARNING] Evaluation TIMED OUT for Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}")

#         print(f"[Eval Result] Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}: {result}")
#         return result


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

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {config.num_cpu_devices} CPUs; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size, f"Current batch size {len(curr_work_batch)} is greater than the number of CPUs {batch_size}"

            with mp.Pool(batch_size) as pool:

                work_args = [
                    (
                        WorkArgs(
                            problem_id=p_id,
                            sample_id=s_idx,
                            device=torch.device("cpu"),
                            # device=torch.device(f"cuda:{i%batch_size}"),
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
                        results.append((problem_id, sample_id, None))
                    
                        remove_cache_dir(config.kernel_eval_build_dir, config.run_name, problem_id, sample_id) # type: ignore
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        results.append((problem_id, sample_id, None))
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
    }
    
    # Write updated results back to file
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f)

# def single_eval_example(config: EvalConfig, curr_level_dataset: list[str], run_dir: str, eval_file_path ):
#     device = torch.device("cuda:0")
#     example_work = WorkArgs(problem_id=1, sample_id=0, device=device)
#     # example_eval_result = evaluate_single_sample(example_work, config, curr_level_dataset, run_dir)
#     example_eval_result = cuda_single_eval_wrapper(example_work, config, curr_level_dataset, run_dir)
#     print(example_eval_result)
#     if not check_if_eval_exists_local(1, 0, eval_file_path):
#         add_to_eval_results_file(1, 0, example_eval_result, eval_file_path)

KERNEL_BENCH_PATH = "/code/LLM4HPCTransCompile/EvalEngine/torch_functionals"
def construct_problem_dataset_from_problem_dir(problem_dir: str) -> list[str]:
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
    return DATASET


def construct_kernelbench_dataset(level: int) -> list[str]:
    return construct_problem_dataset_from_problem_dir(
        os.path.join(KERNEL_BENCH_PATH, f"level{level}")
    )

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Batch Eval Samples from Particular Run
    Store Eval Results in specified eval results file
    """
    print(f"Starting Batch Eval with config: {config}")
    
    # # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    curr_level_dataset = construct_kernelbench_dataset(config.level) # type: ignore
    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}" # type: ignore
        problem_id_range = range(config.subset[0], config.subset[1]) # type: ignore

    print(f"Evaluating 1 sample each for level {config.level} problems: {problem_id_range}")

    run_dir = os.path.join(config.runs_dir, config.run_name) # type: ignore
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    print(num_problems_in_level, run_dir, eval_file_path)

    # To Debug
    # single_eval_example(config, curr_level_dataset, run_dir, eval_file_path)

    total_work = []
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        sample_id = 0 # only evaluate 1 sample for now
        if not check_if_eval_exists_local(problem_id, sample_id, eval_file_path):
            total_work.append((problem_id, sample_id))

    print(f"Start evaluation on {len(total_work)} unevaluated samples in range: {problem_id_range}")
    # # Build Cache on CPU as that is faster
    # if config.build_cache:
    #     compile.batch_compile(total_work, config.to_dict())

    # Batch Eval on CPUs
    batch_eval(total_work, config, curr_level_dataset, run_dir, eval_file_path)


if __name__ == "__main__":
    main()