import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class MDocGraphState(TypedDict, total=False):
    question: str
    texts: list[str]
    images: list[str]
    general_response: str
    text_reflection: str
    image_reflection: str
    text_response: str
    image_response: str
    final_answer: Any
    agent_details: dict[str, Any]


def parse_reflections(raw_reflection):
    start_index = raw_reflection.find("{")
    end_index = raw_reflection.find("}") + 1
    if start_index < 0 or end_index <= start_index:
        return "", ""
    try:
        critical_info = json.loads(raw_reflection[start_index:end_index])
    except Exception as e:
        print(e)
        return "", ""
    return critical_info.get("text", ""), critical_info.get("image", "")


def build_mdoc_graph(system):
    graph = StateGraph(MDocGraphState)

    def general_node(state):
        general_agent = system.agents[-1]
        general_response, _ = general_agent.predict(
            state["question"],
            state.get("texts"),
            state.get("images"),
            with_sys_prompt=True,
        )
        return {
            "general_response": general_response,
        }

    def reflect_node(state):
        general_agent = system.agents[-1]
        raw_reflection = general_agent.self_reflect(
            prompt=general_agent.config.agent.critical_prompt,
            add_to_message=False,
        )
        text_reflection, image_reflection = parse_reflections(raw_reflection)
        return {
            "text_reflection": text_reflection,
            "image_reflection": image_reflection,
        }

    def text_node(state):
        text_agent = system.agents[1]
        reflect_prompt = "\nYou may use the given clue:\n"
        text_response, _ = text_agent.predict(
            state["question"] + reflect_prompt + state.get("text_reflection", ""),
            texts=state.get("texts"),
            images=None,
            with_sys_prompt=True,
        )
        return {"text_response": text_response}

    def image_node(state):
        image_agent = system.agents[0]
        reflect_prompt = "\nYou may use the given clue:\n"
        image_response, _ = image_agent.predict(
            state["question"] + reflect_prompt + state.get("image_reflection", ""),
            texts=None,
            images=state.get("images"),
            with_sys_prompt=True,
        )
        return {"image_response": image_response}

    def sum_node(state):
        all_messages = "General Agent:\n" + state.get("general_response", "") + "\n"
        all_messages += "Text Agent:\n" + state.get("text_response", "") + "\n"
        all_messages += "Image Agent:\n" + state.get("image_response", "") + "\n"
        final_ans, _ = system.sum(all_messages)
        agent_details = {
            "General Agent": state.get("general_response", ""),
            "Text Agent": state.get("text_response", ""),
            "Image Agent": state.get("image_response", ""),
            "Sum Agent": final_ans,
        }
        return {
            "final_answer": final_ans,
            "agent_details": agent_details,
        }

    graph.add_node("general", general_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("text", text_node)
    graph.add_node("image", image_node)
    graph.add_node("sum", sum_node)

    graph.add_edge(START, "general")
    graph.add_edge("general", "reflect")
    graph.add_edge("reflect", "text")
    graph.add_edge("reflect", "image")
    graph.add_edge("text", "sum")
    graph.add_edge("image", "sum")
    graph.add_edge("sum", END)

    return graph.compile()
