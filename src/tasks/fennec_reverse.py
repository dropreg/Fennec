from data.eval_event import EvalEvent
from .registry import auto_register
from .meta import Task
from actions.fennec_reverse import FennecReverseBranchAction, FennecReverseScoringAction


@auto_register("fennec_reverse")
class FennecReverse(Task):
    task_name = "fennec_reverse"

    def __init__(self, config, task_func, db_server, llm_server) -> None:
        super().__init__(config, task_func, db_server, llm_server)

        self.logger.info("Init Task {}".format(self.task_name))

    def annotation(self):
        pass

    def evaluation(self, eval_event: EvalEvent, server, eval_model):
        pass

    def generation(self, eval_event: EvalEvent, server, eval_model):
        dialogue = eval_event.get_dialogue()
        session_id = dialogue.get_session_id()
        meta_info = dialogue.get_meta_info()
        
        if self.task_func == "pairwise_gen_func":
            turn_idx = meta_info["turn"] - 1
            
            if "info" not in meta_info:
                result2criterion = "Given a [User Query] and the subsequent responses (A and B) provided by two AI assistants, along with an in-depth and impartial [Evaluation Result] of these responses, please formulate an [Evaluation Criterion] that demonstrates a high correlation with the relevance of the user's query and assistant's responses.\n***\n[User Query]:\n{query}\n***\n[The Start of Assistant 1's Response]:\n{response1}\n[The End of Assistant 1's Response]\n***\n[The Start of Assistant 2's Response]:\n{response2}\n[The End of Assistant 2's Response]\n***\n[The Start of Evaluation Result]:\n{result}\n[The End of Evaluation Result]\n***\nPlease return [Evaluation Criterion]:\n"
                
                result2score = "Given a [User Query] and the subsequent responses (A and B) provided by two AI assistants, along with a comprehensive and unbiased [Evaluation Criteria] and [Evaluation Result] for these responses. Please contemplate and summarize the scoring scales associated with these [Evaluation Results] to develop a rational and highly relevant [Scoring Guideline].\n***\n[User Query]:\n{query}\n***\n[The Start of Assistant 1's Response]:\n{response1}\n[The End of Assistant 1's Response]\n***\n[The Start of Assistant 2's Response]:\n{response2}\n[The End of Assistant 2's Response]\n***\n[The Start of Evaluation Result]:\n{result}\n[The End of Evaluation Result]\n***\nPlease return [Scoring Guideline]:\n"

                judgment = meta_info["judgment"][0]
                
                if "final decision" in judgment:
                    result = judgment_result = "Explanation:" + judgment
                else:
                    score = judgment.split("\n")[0].split()
                    judgment_info = "".join(judgment.split("\n")[1:])
                    judgment_result = "Assistant 1's Response Score: {score_a}\nExplanation: {judgement}\nAssistant 2's Response Score: {score_b}\nExplanation: {judgement}".format(
                        score_a=int(float(score[0])),
                        score_b=int(float(score[1])),
                        judgement=judgment_info,
                    )

                    result = "Assistant 1's Response Score: {score_a}\nAssistant 2's Response Score: {score_b}\nExplanation: {judgement}".format(
                        score_a=int(float(score[0])),
                        score_b=int(float(score[1])),
                        judgement=judgment_info,
                    )
                eval_event.update_memory(
                    "memory",
                    "result",
                    result,
                )
                query = dialogue.get_query_by_idx(0)["content"].split()
                response1 = dialogue.get_pairwise_response_by_idx(0, "model_a")[
                    "content"
                ]
                response2 = dialogue.get_pairwise_response_by_idx(0, "model_b")[
                    "content"
                ]
                candi_branch = result2criterion.format(
                    query=query,
                    response1=response1,
                    response2=response2,
                    result=judgment_result,
                )
                frb_action_feedback = self.run_action(
                    FennecReverseBranchAction.action_name,
                    session_id,
                    turn_idx=str("turn{}".format(turn_idx)),
                    action_input={
                        "dialogue": dialogue,
                        "candi_branch": [candi_branch],
                        "server": server,
                        "eval_model": eval_model,
                    },
                )
                eval_event.update_memory(
                    self.task_name,
                    FennecReverseBranchAction.action_name,
                    frb_action_feedback,
                )
                
                candi_scoring = result2score.format(
                    query=query,
                    response1=response1,
                    response2=response2,
                    result=judgment_result,
                    criteria=frb_action_feedback[str("turn{}".format(turn_idx))][
                        "branch_list"
                    ][0],
                )
                frs_action_feedback = self.run_action(
                    FennecReverseScoringAction.action_name,
                    session_id,
                    turn_idx=str("turn{}".format(turn_idx)),
                    action_input={
                        "dialogue": dialogue,
                        "candi_scoring": [candi_scoring],
                        "server": server,
                        "eval_model": eval_model,
                    },
                )
                eval_event.update_memory(
                    self.task_name,
                    FennecReverseScoringAction.action_name,
                    frs_action_feedback,
                )
            else:
                frb_action_feedback = self.run_action(
                    FennecReverseBranchAction.action_name,
                    session_id,
                    turn_idx=str("turn{}".format(turn_idx)),
                    action_input={
                        "dialogue": dialogue,
                        "candi_branch": meta_info["info"]["candi_branch"],
                        "server": server,
                        "eval_model": eval_model,
                    },
                )
                eval_event.update_memory(
                    self.task_name,
                    FennecReverseBranchAction.action_name,
                    frb_action_feedback,
                )

                frs_action_feedback = self.run_action(
                    FennecReverseScoringAction.action_name,
                    session_id,
                    turn_idx=str("turn{}".format(turn_idx)),
                    action_input={
                        "dialogue": dialogue,
                        "candi_scoring": meta_info["info"]["candi_scoring"],
                        "server": server,
                        "eval_model": eval_model,
                    },
                )
                eval_event.update_memory(
                    self.task_name,
                    FennecReverseScoringAction.action_name,
                    frs_action_feedback,
                )

            self.logger.info(
                "FennecReverse Execute {} with Turn {}".format(self.task_name, turn_idx)
            )
        else:
            raise Exception("Not Support Task Function {}".format(self.task_func))
