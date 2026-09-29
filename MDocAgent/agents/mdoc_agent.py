
from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
        self.graph = None
        self.use_langgraph = getattr(config, "use_langgraph", True)
        if self.use_langgraph:
            try:
                from agents.mdoc_graph import build_mdoc_graph

                self.graph = build_mdoc_graph(self)
            except Exception as e:
                print(f"LangGraph disabled: {e}")
                self.use_langgraph = False
    
    def predict(self, question, texts, images):
        if self.use_langgraph and self.graph is not None:
            state = {
                "question": question,
                "texts": texts,
                "images": images,
            }
            graph_config = self._graph_tracing_config()
            if graph_config:
                result = self.graph.invoke(state, config=graph_config)
            else:
                result = self.graph.invoke(state)
            return result["final_answer"], result["agent_details"]
        return self._predict_legacy(question, texts, images)

    def _graph_tracing_config(self):
        tracing = getattr(getattr(self, "config", None), "tracing", None)
        if not bool(getattr(tracing, "enabled", False)):
            return {}
        project = str(getattr(tracing, "project", "mhier-agent"))
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        config = {}
        run_name = getattr(tracing, "run_name", None)
        if run_name:
            config["run_name"] = str(run_name)
        tags = list(getattr(tracing, "tags", []))
        if tags:
            config["tags"] = tags
        metadata = dict(getattr(tracing, "metadata", {}))
        if metadata:
            config["metadata"] = metadata
        return config

    def _predict_legacy(self, question, texts, images):
        general_agent = self.agents[-1]
        general_response, messages = general_agent.predict(question, texts, images, with_sys_prompt=True)
        critical_info = general_agent.self_reflect(prompt = general_agent.config.agent.critical_prompt, add_to_message=False)

        start_index = critical_info.find('{')
        end_index = critical_info.find('}') + 1
        critical_info = critical_info[start_index:end_index]
        text_reflection = ""
        image_reflection = ""
        try:
            critical_info = json.loads(critical_info)
            text_reflection = critical_info.get("text", "")
            image_reflection = critical_info.get("image", "")
        except Exception as e:
            print(e)

        text_agent = self.agents[1]
        image_agent = self.agents[0]

        agent_details = {
            "General Agent": general_response,
        }

        relect_prompt = "\nYou may use the given clue:\n"
        text_response, messages = text_agent.predict(question + relect_prompt +text_reflection, texts = texts, images = None, with_sys_prompt=True)
        agent_details["Text Agent"] = text_response
        image_response, messages = image_agent.predict(question + relect_prompt +image_reflection, texts = None, images = images, with_sys_prompt=True)
        agent_details["Image Agent"] = image_response

        all_messages = "General Agent:\n" + general_response + "\n"
        all_messages += "Text Agent:\n" + text_response + "\n"
        all_messages += "Image Agent:\n" + image_response + "\n"

        final_ans, final_messages = self.sum(all_messages)
        agent_details["Sum Agent"] = final_ans

        return final_ans, agent_details
