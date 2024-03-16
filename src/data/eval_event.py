from data.dialogue import Dialogue
from utils import common_utils


class EvalEvent:
    def __init__(self, dialogue, queue_id) -> None:
        self.logger = common_utils.get_loguru()

        self.dialogue = dialogue
        self.queue_id = queue_id
        self.working_memories = {}

    def get_queue_id(self):
        return self.queue_id

    def get_dialogue(self) -> Dialogue:
        return self.dialogue

    def get_working_memories(self):
        return self.working_memories

    def not_empty(self):
        return len(self.working_memories) > 0

    def clear(self):
        self.working_memories = {}

    def get_memories(self, task_name, action_name, turn_idx=None):
        if turn_idx is not None:
            return self.working_memories[task_name][action_name][turn_idx]
        else:
            return self.working_memories[task_name][action_name]

    def update_memory(self, task_name, action_name, feedback):
        if task_name not in self.working_memories:
            self.working_memories[task_name] = {}

        if action_name not in self.working_memories[task_name]:
            self.working_memories[task_name][action_name] = {}

        self.logger.info(
            "Update EvalEvent Memories for Task_name = {}, action_name = {}".format(
                task_name, action_name
            )
        )
        self.working_memories[task_name][action_name] = feedback
