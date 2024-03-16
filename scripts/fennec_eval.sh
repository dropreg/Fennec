

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


python src/auto_eval.py \
-c scripts/gen_conf/fennec_bench_v2_gen.yaml \
-a -p 64 \
