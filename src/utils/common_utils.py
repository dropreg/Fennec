from loguru import logger
import sys
import time


lock_loguru = None


def get_loguru():
    global lock_loguru
    if lock_loguru is None:
        logger.remove()
        logger.add(sys.stdout, level="INFO", enqueue=True)
        logger.add("logfiles/log.txt", rotation="10 MB", level="INFO", enqueue=True)
        lock_loguru = logger
    return lock_loguru


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_whole_time():
    return time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime())


def get_mt_bench_eval_model():
    return {
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-instant-v1",
        "vicuna-33b-v1.3",
        "wizardlm-30b",
        "Llama-2-70b-chat",
        "Llama-2-13b-chat",
        "vicuna-13b-v1.3",
        "wizardlm-13b",
        "alpaca-13b",
        "chatglm-6b",
        "rwkv-4-raven-14b",
        "dolly-v2-12b",
        "llama-13b",
    }
