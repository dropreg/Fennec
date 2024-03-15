from .registry import auto_register
from .meta import Bench
from data.dialogue import Dialogue, PairwiseDialogue
import json
import os
import pdb


@auto_register("mt_bench")
class MTbench(Bench):
    bench_name = "mt_bench"

    def __init__(self, config) -> None:
        super().__init__(config)

        self.datasets = self.config["datasets"]

        self.q2c = {}
        for line in open(self.config["question"]["raw_file"]).readlines():
            json_data = json.loads(line)
            self.q2c[json_data["question_id"]] = json_data["category"]
    
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
            for line in open(raw_file).readlines():
                json_data = json.loads(line)
                if data_type == "pairwise":
                    
                    if "turn0" in format_file and json_data["turn"] != 1:
                        continue
                    
                    meta_info = {
                        "question_id": json_data["question_id"],
                        "model_a": json_data["model_a"],
                        "model_b": json_data["model_b"],
                        "category": self.q2c[json_data["question_id"]],
                        "judge": [json_data["winner"]],
                        "judgment": [""],
                        "turn": json_data["turn"],
                        "score": "",
                    }
                    meta_info.update({"reference": []})
                    fw.writelines(
                        json.dumps(
                            {
                                "meta_info": meta_info,
                                "conversation_a": json_data["conversation_a"],
                                "conversation_b": json_data["conversation_b"],
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
                dialogue = Dialogue()
                for idx in range(len(json_data["conversation"]) // 2):
                    query = json_data["conversation"][idx * 2]
                    response = json_data["conversation"][idx * 2 + 1]
                    dialogue.add_turn_item(
                        query, response, None, json_data["meta_info"]["model"]
                    )
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
