<p align="center">
  <img src="fennec_logo2.png" width=256px>
</p>

<h1 align="center"> Fennec: Fine-grained Language Model Evaluation and Correction Extended through Branching and Bridging </h1>

## Table of contents

- [Introduction](#Introduction)
- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Model](#model)
  - [Usage](#usage)
- [Data](#data)
  - [Training Data](#training-data)
  - [Evaluation Benchmark](#evaluation-benchmark)

## Introduction

Fennec 可以针对 “对话数据” 进行有效评估。

| 这里的对话数据是指为了解决 “复杂的人类意图为目的” 的对话数据，而非 “闲聊性” 对话。

## Quick start

当前代码库中包括了如何对评估模型的 “构建训练数据” 和 “推理” 的功能。

### Setup

### Usage

通过 Vllm 可以直接运行评估模型服务，然后通过Post请求的方式来执行推理。

```bash
bash scripts/run_vllm_server.sh
bash scripts/fennec_eval.sh
```

Fennec 支持多种推理模式，并且通过配置文件的的方式来进行管理：

+ fennec 针对对话数据进行评估，并依次生成：“评估标准”、“打分指导”、“评判结果”、并针对低分的内容进行修正。
+ fennec_reverse 为开源数据中的 “评判结果” 生成相应的 “评估标准” 和 “打分指导”。
+ single_eval 通过为 “单个回复” 提供判决结果。
+ pairwise_eval 通过为 “成对回复” 提供判决结果。
+ prometheus_eval 评估模型 Prometheus。
+ bsm_eval 评估模型 BSM。

## Data

Fennec 可以从头构建数据包括以下来源：[benchmark.yaml](conf/benchmark.yaml)

+ autoj_bench 使用 Auto-J 中数据进行训练模型和进行相应评估。

```bash
bash scripts/download.sh
python src/prepare_bench.py -c conf/benchmark.yaml -b autoj_bench
```

### Training Data

Fennec Dataset

Fennec-Bridging Dataset

### Evaluation Benchmark

Auto-J

PandaLM

MT-Bench Human Judgment (Turn0)

Correction

System Rank

TODO:

1. 通过构建不同难度的数据来考察模型训练能力的区别：如果没有见过低分的回复，能否给出准确的回答。收集数据
2. 针对 response 来找合适关系。
4. 扩展数据大小、轮数、领域数据。

