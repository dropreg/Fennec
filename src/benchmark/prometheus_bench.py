from .registry import auto_register
from .meta import Bench
from data.dialogue import Dialogue
import json
import os
import pdb


@auto_register("prometheus_bench")
class PrometheusBench(Bench):
    bench_name = "prometheus_bench"

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

    def filter(self, content):
        if "\n***\n[Query]:" in content:
            content = content.replace("\n***\n[Query]:", "")
        if "\n***\n[Response 1]:" in content:
            content = content.replace("\n***\n[Response 1]:", "")
        if "\n***\n[Response 2]:" in content:
            content = content.replace("\n***\n[Response 2]:", "")
        return content.strip()

    def build_format_data(self, data_type, raw_file, format_file):
        with open(format_file, "w") as fw:
            line_idx = 0
            
            for json_data in json.load(open(raw_file)):
                line_idx += 1
                if data_type == "single":
                    meta_info = {
                        "question_id": line_idx,
                        "model": "gpt-4",
                        "category": "generation",
                        "judge": [json_data['orig_score']],
                        "judge_type": "gpt-4",
                        "judgment": [json_data['orig_feedback']],
                        "turn": 1,
                        "score": json_data['orig_score'],
                        "instruction": json_data['instruction'],
                    }
                    
                    conversation = [
                        {
                            "content": "Instruction: {} Input: {}".format(json_data['orig_instruction'], json_data['input']),
                            "role": "user",
                        },
                        {
                            "content": json_data['orig_response'],
                            "role": "assistant",
                        },
                    ]
                    
                    meta_info.update({"reference": json_data["orig_reference_answer"]})
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
            else:
                raise Exception("Not Support")
            dialogue.set_meta_info(json_data["meta_info"])
            index_field = json_data["meta_info"]["category"]
            count += 1

            zookeeper = self.get_zookeeper()
            db_handler.save_dialogue2db(dialogue, index_field, zookeeper, "0")
        self.logger.info(
            "prepare prometheus_bench dialogue handle data = {}, current data = {}.".format(
                count, db_handler.get_dialogue_size()
            )
        )
