import uuid
import json


class EvalBranch:

    def __init__(self, criteria, scoring, feedback, correction, branch_id=None) -> None:
        
        if branch_id:
            self.branch_id = branch_id
        else:
            self.branch_id = uuid.uuid4().hex

        self.criteria = criteria
        self.scoring = scoring
        self.feedback = feedback
        self.correction = correction

    def dumps(self):
        return {
            "criteria": self.criteria,
            "scoring": self.scoring,
            "feedback": self.feedback,
            "correction": self.correction
        }

    def loads(self, json_data):
        self.criteria = json_data["criteria"]
        self.scoring = json_data["scoring"]
        self.feedback = json_data["feedback"]
        self.correction = json_data["correction"]


class EvalInfo:

    def __init__(self, eval_id=None) -> None:

        if eval_id:
            self.eval_id = eval_id
        else:
            self.eval_id = uuid.uuid4().hex

        self.branchs = []
        self.meta = {}

    def dumps(self):
        data = {
            "eval_id": self.eval_id,
            "meta": self.meta,
            "branchs": [b.dumps() for b in self.branchs],
        }
        return json.dumps(data, ensure_ascii=False)

    def loads(self, json_data):
        self.eval_id = json_data["eval_id"]
        self.meta = json_data["meta"]
        for json_branch in json_data["branchs"]:
            branch = EvalBranch("", "")
            branch.loads(json_branch)
            self.branchs.append(branch)
