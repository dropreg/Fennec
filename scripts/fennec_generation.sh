

export CUDA_VISIBLE_DEVICES=0

python src/auto_eval.py \
-c scripts/gen_conf/autoj_fennec.yaml \
-a -p 2

# python src/auto_eval.py \
# -c scripts/gen_conf/autoj_fennec_chatbot_arena.yaml \
# -a -p 12

# python src/auto_eval.py \
# -c scripts/gen_conf/autoj_fennec_reverse.yaml \
# -a -p 64 \

# python src/auto_eval.py \
# -c scripts/gen_conf/autoj_fennec_reverse_autoj.yaml \
# -a -p 64 \
