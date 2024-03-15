from utils import common_utils
from data.dialogue import Dialogue, PairwiseDialogue, ReferenceDialogue
import json


class DialogueHandler:
    def __init__(self, db_server) -> None:
        self.logger = common_utils.get_loguru()
        self.db = db_server

    def get_dialogue_size(self):
        return self.db.get_dialogue_size()

    def save_dialogue2db(self, dialogue, index_field, zookeeper, state):
        session_id = dialogue.get_session_id()
        data = dialogue.dumps()
        self.db.save_dialogue(session_id, data, index_field, zookeeper, state)

    def update_dialogue2db(self, dialogue):
        session_id = dialogue.get_session_id()
        data = dialogue.dumps()
        self.db.update_dialogue(session_id, data)

    def load_all_dialogue(self):
        dialogue_list = []
        for json_data_str in self.db.get_all_dialogue():
            json_data = json.loads(json_data_str["data"])

            meta_type = json_data["meta_type"]
            if meta_type == "PairwiseDialogue":
                dialogue = PairwiseDialogue()
            elif meta_type == "ReferenceDialogue":
                dialogue = ReferenceDialogue()
            elif meta_type == "Dialogue":
                dialogue = Dialogue()
            else:
                raise Exception("Not Supprot {}".format(meta_type))

            dialogue.loads(json_data)
            dialogue_list.append(dialogue)

        return dialogue_list
