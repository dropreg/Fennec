

# Pairwise Evaluation
# scripts/eval_conf/pairwise/autoj_bench_evalp_1392_fennec_pairwise_eval.yaml \
# scripts/eval_conf/pairwise/pandalm_bench_testset_fennec_pairwise_eval.yaml \
# scripts/eval_conf/pairwise/mt_bench_human_pair_3355_turn0_fennec_pairwise_eval.yaml \

# Single Evaluation
# scripts/eval_conf/single/autoj_bench_evalp_1392_fennec_pairwise_single_eval.yaml \
# scripts/eval_conf/single/pandalm_bench_testset_fennec_pairwise_single_eval.yaml \
# scripts/eval_conf/single/mt_bench_human_pair_3355_turn0_fennec_pairwise_single_eval.yaml \

# python src/auto_eval.py \
# -c scripts/eval_conf/pairwise/autoj_bench_evalp_1392_fennec_pairwise_eval.yaml \
# -a -p 64 \

# python src/auto_eval.py \
# -c scripts/gen_conf/autoj_fennec.yaml \
# -a -p 2 \
# export OPENBLAS_NUM_THREADS=4
# export GOTO_NUM_THREADS=4
# export OMP_NUM_THREADS=4

python src/auto_eval.py \
-c scripts/eval_conf/correction_bench_autoj_fennec_pairwise_eval.yaml \
-a -p 4 \


# python src/auto_eval.py \
# -c scripts/gen_conf/autoj_fennec_0426.yaml \
# -a -p 64 \