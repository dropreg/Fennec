from .registry import auto_register
from .meta import Bench
from data.dialogue import Dialogue, PairwiseDialogue
import json
import os
import pdb


@auto_register("metatool_bench")
class MetaToolbench(Bench):
    bench_name = "metatool_bench"

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
            json_data_list = json.load(open(raw_file))
            question_id = 0
            for json_data in json_data_list:
                question_id += 1
                if "tool_awareness" in format_file and data_type == "single":

                    example_begin = json_data["thought_prompt"].find("[Examples Start]")
                    example_end = json_data["thought_prompt"].find("[Examples End]")
                    example = json_data["thought_prompt"][example_begin:example_end].replace("[Examples Start]", "")

                    meta_info = {
                        "question_id": question_id,
                        "model": "human",
                        "category": "",
                        "judge": [json_data['label']],
                        "judgment": [json_data['tool'] if "tool" in json_data else ""],
                        "turn": 1,
                        "example": example,
                    }
            
                    conversation = [
                        {
                            "content": json_data['query'],
                            "role": "user",
                        },
                        {
                            "content": "",
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
