from persistence.dialogue_handler import DialogueHandler
from db.sqlalchemy_db import DialogueAlchemyDatabase
from pipe.eval_pipe import EvalPipe
from async_run.eval_queue import AsyncEvalQueue
from utils import common_utils
from config.auto_eval_config import AutoEvalConfig
import argparse


def process_func(pipe, data_idx, data, thread_idx):
    return pipe.run(data_idx, data, thread_idx)


def async_detect(parallel_number, dialogue_list, eval_pipe):
    async_eval_queue = AsyncEvalQueue(parallel_number)

    for dialogue_idx, dialogue in enumerate(dialogue_list):
        async_eval_queue.push((dialogue_idx, dialogue))

    async_eval_queue.run(process_func, eval_pipe)


def run(async_run, config_file, parallel_number, batchsize):
    logger = common_utils.get_loguru()
    logger.info(
        "Auto Evaluation or Gernation with {}".format(
            "Parallel Execution" if async_run else "Step-by-Step Execution"
        )
    )
    
    auto_eval_config = AutoEvalConfig(config_file)
    db_name = auto_eval_config.get_dialogue_db()
    
    db_server = DialogueAlchemyDatabase(db_name)
    handler = DialogueHandler(db_server)
    
    dialogue_list = handler.load_all_dialogue()
    logger.info("Load data = {} from {}".format(len(dialogue_list), db_name))

    eval_pipe = EvalPipe(auto_eval_config)

    if async_run:
        async_detect(parallel_number, dialogue_list, eval_pipe)
    else:
        for dialogue_idx, dialogue in enumerate(dialogue_list):
            eval_pipe.run(dialogue_idx, dialogue)

    eval_pipe.serilaize_eval()

    logger.info("Auto Execution End.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="auto evalutation and generation")
    parser.add_argument(
        "-a", "--async-run", action="store_true", help="run in async mode"
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=1,
        help="The batch size for model inference",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="conf/auto_config.yaml",
        help="The file of configuration.",
    )
    parser.add_argument(
        "-p",
        "--parallel-number",
        type=int,
        default=1,
        help="The number of concurrent API calls.",
    )
    args = parser.parse_args()
    run(args.async_run, args.config_file, args.parallel_number, args.batchsize)
