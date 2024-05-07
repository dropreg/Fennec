from .registry import auto_register
from .meta import Bench
from data.dialogue import PairwiseDialogue
import json
import os


@auto_register("autoj_bench")
class AutoJbench(Bench):
    bench_name = "autoj_bench"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.datasets = self.config["datasets"]

    def prepare(self):
        for _, dataset in self.datasets.items():
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
            for line in open(raw_file).readlines():
                line_idx += 1
                json_data = json.loads(line)

                if data_type == "pairwise":
                    if "fennec_bench_v2" in format_file:
                        meta_info = {
                            "question_id": line_idx,
                            "model_a": json_data["model_1"],
                            "model_b": json_data["model_2"],
                            "category": json_data["source_file"],
                            "judge": [""],
                            "judge_type": "pairwise",
                            "judgment": [""],
                            "turn": 1,
                            "score": "",
                            "context": json_data["context"],
                        }

                        conversation_a = [
                            {
                                "content": self.filter(json_data["query"]),
                                "role": "user",
                            },
                            {
                                "content": self.filter(json_data["response_1"]),
                                "role": "assistant",
                            },
                        ]
                        conversation_b = [
                            {
                                "content": self.filter(json_data["query"]),
                                "role": "user",
                            },
                            {
                                "content": self.filter(json_data["response_2"]),
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
                    elif "train" in format_file:
                        meta_info = {
                            "question_id": line_idx,
                            "model_a": json_data["source_dataset"],
                            "model_b": json_data["source_dataset"],
                            "category": json_data["scenario"],
                            "judge": [json_data["gt_label"], json_data["pred_label"]],
                            "judge_type": "pairwise",
                            "judgment": [json_data["target_output"]],
                            "turn": 1,
                            "score": "",
                        }

                        prompt = json_data["usrmsg"]
                        q_pos = prompt.find("\n***\n[Query]")
                        r1_pos = prompt.find("\n***\n[Response 1]")
                        r2_pos = prompt.find("\n***\n[Response 2]")
                        end_pos = prompt.find("\n***\n[END DATA]")

                        conversation_a = [
                            {
                                "content": self.filter(prompt[q_pos:r1_pos]),
                                "role": "user",
                            },
                            {
                                "content": self.filter(prompt[r1_pos:r2_pos]),
                                "role": "assistant",
                            },
                        ]
                        conversation_b = [
                            {
                                "content": self.filter(prompt[q_pos:r1_pos]),
                                "role": "user",
                            },
                            {
                                "content": self.filter(prompt[r2_pos:end_pos]),
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

                    elif "test" in format_file:
                        meta_info = {
                            "question_id": line_idx,
                            "model_a": "",
                            "model_b": "",
                            "category": json_data["scenario"],
                            "judge": [json_data["label"]],
                            "judge_type": "pairwise",
                            "judgment": [""],
                            "turn": 1,
                            "score": "",
                        }

                        conversation_a = [
                            {
                                "content": self.filter(json_data["prompt"]),
                                "role": "user",
                            },
                            {
                                "content": self.filter(json_data["response 1"]),
                                "role": "assistant",
                            },
                        ]
                        conversation_b = [
                            {
                                "content": self.filter(json_data["prompt"]),
                                "role": "user",
                            },
                            {
                                "content": self.filter(json_data["response 2"]),
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
