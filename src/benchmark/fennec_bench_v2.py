from .registry import auto_register
from .meta import Bench
from data.dialogue import Dialogue, PairwiseDialogue
import json
import os
import pdb


@auto_register("fennec_bench_v2")
class FennecBenchV2bench(Bench):
    bench_name = "fennec_bench_v2"

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
            
            question_id = 0
            for line in open(raw_file).readlines():
                json_data = json.loads(line)
                question_id += 1
                if data_type == "single":
                    
                    meta_info = {
                        "question_id": question_id,
                        "model": "",
                        "category": json_data['class'],
                        "judge": "",
                        "judgment": [""],
                        "turn": 1,
                        "source_file": json_data['source'],
                        "context": json_data['context'] if "context" in json_data else ""
                    }
                    
                    if "input" in json_data and json_data['input']:
                        conversation = [
                            {
                                "content": "[Instruction]:{}\n[Input]:{}".format(json_data['query'], json_data['input']),
                                "role": "user",
                            },
                            {
                                "content": json_data['response'],
                                "role": "assistant",
                            },
                        ]
                    else:
                        conversation = [
                            {
                                "content": json_data['query'],
                                "role": "user",
                            },
                            {
                                "content": json_data['response'],
                                "role": "assistant",
                            },
                        ]
                    meta_info.update({"reference": []})
                    fw.writelines(
                        json.dumps(
                            {
                                "meta_info": meta_info,
                                "conversation": conversation,
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
            if data_type == "single":
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
