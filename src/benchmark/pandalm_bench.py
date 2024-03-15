import pdb
from .registry import auto_register
from .meta import Bench
from data.dialogue import PairwiseDialogue
import json
import os


@auto_register("pandalm_bench")
class PandaLMbench(Bench):
    bench_name = "pandalm_bench"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.datasets = self.config["datasets"]

    def prepare(self):
        for dataset_id, dataset in self.datasets.items():
            raw_file = dataset["raw_file"]
            format_file = dataset["format_file"]
            db_file = dataset["db_file"]
            data_type = dataset["type"]

            if not os.path.exists(os.path.dirname(format_file)):
                os.makedirs(os.path.dirname(format_file))
            self.build_format_data(data_type, raw_file, format_file)
            if not os.path.exists(os.path.dirname(db_file)):
                os.makedirs(os.path.dirname(db_file))
            self.build_db(data_type, format_file, db_file)

    def build_format_data(self, data_type, raw_file, format_file):
        with open(format_file, "w") as fw:
            line_idx = 0

            for json_data in json.load(open(raw_file)):
                line_idx += 1

                if data_type == "pairwise":
                    meta_info = {
                        "question_id": line_idx,
                        "model_a": json_data["cmp_key"].split("_")[0],
                        "model_b": json_data["cmp_key"].split("_")[1],
                        "category": json_data["motivation_app"],
                        "judge": [
                            json_data["annotator1"],
                            json_data["annotator2"],
                            json_data["annotator3"],
                        ],
                        "judge_type": "pairwise",
                        "judgment": ["", "", ""],
                        "turn": 1,
                        "score": "",
                    }

                    conversation_a = [
                        {
                            "content": json_data["instruction"] + json_data["input"],
                            "role": "user",
                        },
                        {
                            "content": json_data["response1"],
                            "role": "assistant",
                        },
                    ]
                    conversation_b = [
                        {
                            "content": json_data["instruction"] + json_data["input"],
                            "role": "user",
                        },
                        {
                            "content": json_data["response2"],
                            "role": "assistant",
                        },
                    ]
                    meta_info.update({"reference": []})
                    fw.writelines(
                        json.dumps(
                            {
                                "meta_info": meta_info,
                                "conversation_a": conversation_a,
                                "conversation_b": conversation_b,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    raise Exception("Not Support")
            self.logger.info(
                "Convert Raw Data {} for Format Data {} Data Type = {}".format(
                    raw_file, format_file, data_type
                )
            )

    def build_db(self, data_type, format_file, db_file):
        count = 0

        self.logger.info("init db handle for file {}".format(db_file))
        db_handler = self.load_db_handler(db_file)

        for line in open(format_file).readlines():
            json_data = json.loads(line)
            if data_type == "pairwise":
                dialogue = PairwiseDialogue()
                for idx in range(len(json_data["conversation_a"]) // 2):
                    query = json_data["conversation_a"][idx * 2]
                    response_a = json_data["conversation_a"][idx * 2 + 1]
                    query_b = json_data["conversation_b"][idx * 2]
                    response_b = json_data["conversation_b"][idx * 2 + 1]
                    assert query == query_b
                    pairwise_response = {
                        "model_a": response_a,
                        "model_b": response_b,
                    }
                    dialogue.add_turn_item(query, pairwise_response, None, "")
            else:
                raise Exception("Not Support")
            dialogue.set_meta_info(json_data["meta_info"])
            index_field = json_data["meta_info"]["category"]
            count += 1

            zookeeper = self.get_zookeeper()
            db_handler.save_dialogue2db(dialogue, index_field, zookeeper, "0")
        self.logger.info(
            "prepare_mt_bench_dialogue handle data = {}, current data = {}.".format(
                count, db_handler.get_dialogue_size()
            )
        )
