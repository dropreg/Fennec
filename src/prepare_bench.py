from utils import common_utils
from config.dialogue_config import DialogueConfig
from benchmark.registry import BenchRegistry
import argparse

logger = common_utils.get_loguru()

def prepare(config_file, bench_name):
    config = DialogueConfig(config_file)

    bench_candidates = config.get_bench_candidates()
    if bench_name in bench_candidates:
        bench = BenchRegistry.create_instance(bench_name, config)
        bench.prepare()
    else:
        raise NotImplementedError(
            "Not Support Benchmark = {}, Please Select From {}".format(
                bench_name, bench_candidates
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="auto evalutation and generation")
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="conf/auto_config.yaml",
        help="The file of configuration.",
    )
    parser.add_argument(
        "-b",
        "--bench",
        type=str,
        default="mt_bench",
        help="The name of Benchmark.",
    )

    args = parser.parse_args()
    prepare(args.config_file, args.bench)
