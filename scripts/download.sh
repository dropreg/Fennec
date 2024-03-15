#!/bin/bash

# 1. Download Auto-j Bench Dataset

# autoj_train_file="https://raw.githubusercontent.com/GAIR-NLP/auto-j/main/data/training/pairwise_traindata.jsonl"
# autoj_test_file="https://raw.githubusercontent.com/GAIR-NLP/auto-j/main/data/test/testdata_pairwise.jsonl"

# local_autoj_train_dir="data/fennec_train_data/raw/autoj"
# local_autoj_train_file="pairwise_traindata.jsonl"
# local_autoj_test_dir="data/fennec_eval_data/raw/auto_j_eval_p"
# local_autoj_test_file="testdata_pairwise.jsonl"

# mkdir -p $local_autoj_train_dir
# mkdir -p $local_autoj_test_dir

# curl -o "$local_autoj_train_dir/$local_autoj_train_file" "$autoj_train_file"

# curl -o "$local_autoj_test_dir/$local_autoj_test_file" "$autoj_test_file"

# 2. Download PandaLM Dataset

# pandalm_test_file="https://raw.githubusercontent.com/WeOpenML/PandaLM/main/data/testset-v1.json"

# local_pandalm_test_dir="data/fennec_eval_data/raw/pandalm/"
# local_pandalm_test_file="testdata_pairwise.json"

# mkdir -p $local_pandalm_test_dir

# curl -o "$local_pandalm_test_dir/$local_pandalm_test_file" "$pandalm_test_file"

# 3. Download MT-Bench Human Judge Dataset
# url="https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/human-00000-of-00001-25f4910818759289.parquet?download=true"

