from .registry import auto_register
from .meta import Action


@auto_register("fact_verification_check")
class FactVerificationCheckAction(Action):

    action_name = "fact_verification_check"

    def __init__(self, action_config, llm_server) -> None:
        super().__init__(action_config)
        
        self.llm_server = llm_server

    def execute(self, **action_input):

        action_feedback = {}
        eval_model = action_input["eval_model"] if "eval_model" in action_input else self.eval_model
        result = self.get_result_from_server(
            action_input["query"], 
            action_input["response"], 
            eval_model
        )
        action_feedback["result"] = result["result"]
        action_feedback["thought"] = result["thought"] 
        return action_feedback

    def get_result_from_server(self, query, response, eval_model):

        prompt = fact_prompt.get_fact_verification_check_prompt(query, response, 
                self.task_description, self.common_rules_list, self.result_format)
        result = self.llm_server.request(prompt, eval_model)
        format_result = common_utils.json_format(result)
        return json.loads(format_result)

@auto_register("aurora_query_search")
class AuroraQuerySearch(Action):

    action_name = "aurora_query_search"

    def __init__(self, action_config, llm_server) -> None:
        super().__init__(action_config)
        
        self.llm_server = llm_server

    def execute(self, **action_input):
        action_feedback = {}
        evidence = self.llm_server.request(action_input["query"], eval_model="aurora")
        action_feedback["knowledge"] = "".join(["[证据{}]:{}".format(idx, v) for idx, v in enumerate(evidence)]) if evidence else ""
        return action_feedback

@auto_register("claim_extraction")
class ClaimExtractionAction(Action):

    action_name = "claim_extraction"

    def __init__(self, action_config, llm_server) -> None:
        super().__init__(action_config)
        
        self.llm_server = llm_server

    def execute(self, **action_input):
        
        action_feedback = {}
        eval_model = action_input["eval_model"] if "eval_model" in action_input else self.eval_model
        result = self.get_result_from_server(action_input["response"], eval_model)
        action_feedback["claim"] = {uuid.uuid4().hex:v for v in result}
        return action_feedback

    def get_result_from_server(self, response, eval_model):

        prompt = fact_prompt.get_claim_extraction_prompt(response, 
                self.task_description, self.common_rules_list, self.result_format)
        result = self.llm_server.request(prompt, eval_model)
        format_result = common_utils.json_format(result)
        self.logger.debug(format_result)
        return json.loads(format_result)

@auto_register("evidence_collection")
class EvidenceCollectionAction(Action):

    action_name = "evidence_collection"

    def __init__(self, action_config, llm_server) -> None:
        super().__init__(action_config)
        
        self.llm_server = llm_server

    def execute(self, **action_input):
        
        action_feedback = {}
        eval_model = action_input["eval_model"] if "eval_model" in action_input else self.eval_model
        result = self.get_result_from_server(action_input["claim"], eval_model)

        evidence_list = []
        for result_item in result:
            evidences = self.llm_server.request(result_item, "aurora")
            for evidence in evidences:
                evidence_list.append("[证据{}]: {}".format(len(evidence_list), evidence))
        
        action_feedback["evidence"] = "".join(evidence_list)
        return action_feedback

    def get_result_from_server(self, response, eval_model):

        prompt = fact_prompt.get_evidence_collection_prompt(response, 
                self.task_description, self.common_rules_list, self.result_format)
        result = self.llm_server.request(prompt, eval_model)
        format_result = common_utils.json_format(result)
        self.logger.debug(format_result)
        return json.loads(format_result)

@auto_register("result_verification")
class ResultVerificationAction(Action):

    action_name = "result_verification"

    def __init__(self, action_config, llm_server) -> None:
        super().__init__(action_config)
        
        self.llm_server = llm_server

    def execute(self, **action_input):
        
        action_feedback = {"verification": {}}
        eval_model = action_input["eval_model"] if "eval_model" in action_input else self.eval_model

        if isinstance(action_input["claim"], dict):
            result = self.get_intrinsic_result_from_server(
                action_input["claim"], 
                action_input["evidence"], 
                eval_model,
            )
            for item in result:
                if item['claim_id'] in action_input["claim"]:
                    action_feedback["verification"][item['claim_id']] = {
                        "claim": action_input["claim"][item['claim_id']],
                        "result": item["result"],
                        "thought": item["thought"],
                    }
        else:
            result = self.get_extrinsic_result_from_server(
                action_input["claim"], 
                action_input["evidence"],
                eval_model
            )
            action_feedback["verification"][action_input["claim_id"]] = {
                "claim": action_input["claim"],
                "result": result["result"],
                "thought": result["thought"],
            }
        return action_feedback

    def get_intrinsic_result_from_server(self, claim, evidence, eval_model):

        prompt = fact_prompt.get_intrinsic_result_verification_prompt(claim, evidence, 
                self.task_description, self.common_rules_list, self.result_format)
        result = self.llm_server.request(prompt, eval_model)
        format_result = common_utils.json_format(result)
        self.logger.debug(format_result)
        return json.loads(format_result)
        
    def get_extrinsic_result_from_server(self, claim, evidence, eval_model):

        prompt = fact_prompt.get_extrinsic_result_verification_prompt(claim, evidence, 
                self.task_description, self.common_rules_list, self.result_format)
        result = self.llm_server.request(prompt, eval_model)
        format_result = common_utils.json_format(result)
        self.logger.debug(format_result)
        return json.loads(format_result)
