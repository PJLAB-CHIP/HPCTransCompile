# HPCTransCompile

## Introduction

This repository contains the official implementation of **HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration.**

## Framework Implementation

<img src=".\pictures\framework.png" style="zoom:80%;" />

We propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The result demonstrates that our framework significantly improves CUDA transpilation, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem.

## Project Architecture

We provide two benchmarks, `HPCTransEval` and `KernelBench_c`. You can find them in the corresponding folders.

```
HPCTransCompile/
|-CodeGenEngine # use tvm to generate cuda and cpu high-performance operators
|-EvalEngine # handles the benchmarking and performance analysis of compiled kernels
|-HPCTransEval # HPCTransEval benchmark
|-InferenceEngine # responsible for model inference, translating input code and generating optimized kernels
|-KernelBench_c # KernelBench_c benchmark
|-TrainEngine # includes scripts and modules for training the translation models
|...
```

## Setup

1. Clone the repository

```
git clone https://github.com/PJLAB-CHIP/HPCTransCompile.git
```

2. Install dependencies: In order to do the assessment correctly, you need to download a modified library `tvm` from https://github.com/PJLAB-CHIP/modified_tvm#. (We recommend creating separate virtual environments for the three engines.)

```
cd HPCTransCompile
pip install TrainEngine_environment.txt # for model training
pip install InferenceEngine.txt # for model inference
pip install EvalEngine.txt # for performance analysis
```

## Usage

### CodeGenEngine

We classify operator types into `single operators`, `combined operators` and `model building blocks`. You can generate them through the following scripts (the model building blocks are manually constructed by humans). You can generate different types of code by changing the operator types.

<img src=".\pictures\ops.png" style="zoom:60%;" />

For single operators, run:

```
bash CodeGenEngine/scripts/auto_code_gen.sh
```

For combined operators, run:

```
bash CodeGenEngine/scripts/complex_code_gen.sh
```

### TrainEngine

Training data examples are in the `TrainEngine/dataset`. You can replace them with your own data as needed. Then, run:

```
bash TrainEngine/train.sh
```

If you want to test the impact of different proportions of training datasets on the experimental results, you can modify the "ratio" in the `train_ratio.sh`. Then ,run:

```
bash TrainEngine/train_ratio.sh
```

### InferenceEngine

After the model is trained, the model inference is carried out through `InferenceEngine`(You need to replace the model, level and action in the script with your own settings.). Then, run:

```
bash InferenceEngine/main.sh
```

### EvalEngine

1. To evaluate on `HPCTransEval`, run:

   ```
   bash EvalEngine/eval_HPCTransEval.sh
   ```

2. To evaluate on `KernelBench_c`, run:

   ```
   bash EvalEngine/eval_KernelBench_c.sh
   ```