from utils import common_utils
import json
import uuid


class TaskHandler:
    def __init__(self, db_server) -> None:
        self.logger = common_utils.get_loguru()
        self.db = db_server

    def build_action_db(self, table_name, feild_list):
        self.db.init_table(table_name, feild_list)

    def load_action_feedback(self, session_id, table_name, feild_name):
        action_feedback_dict_serial = self.db.get_feild_by_id(
            table_name, session_id, feild_name
        )

        if len(action_feedback_dict_serial):
            if action_feedback_dict_serial[feild_name] is None:
                return {}
            else:
                action_feedback_dict = json.loads(
                    action_feedback_dict_serial[feild_name]
                )
                self.logger.debug("TaskHandler Load Data {} From DB".format(feild_name))
                return action_feedback_dict
        else:
            self.logger.debug("TaskHandler Data {} Not Found in DB".format(feild_name))
            return None

    def save_action_feedback(
        self, session_id, table_name, feild_name, action_feedback_dict
    ):
        if self.db.exist(table_name, session_id):
            self.update_action_feedback(
                session_id, table_name, feild_name, action_feedback_dict
            )
        else:
            action_feedback_dict_serial = json.dumps(
                action_feedback_dict, ensure_ascii=False
            )
            self.db.save_feild(
                table_name, session_id, feild_name, action_feedback_dict_serial
            )

            self.logger.debug("TaskHandler Save Data {}".format(feild_name))

    def update_action_feedback(
        self, session_id, table_name, feild_name, action_feedback_dict
    ):
        action_feedback_dict_serial = json.dumps(
            action_feedback_dict, ensure_ascii=False
        )
        self.db.update_feild(
            table_name, session_id, feild_name, action_feedback_dict_serial
        )

        self.logger.debug("TaskHandler Update Data {}".format(feild_name))
