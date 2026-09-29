import os
import sys
import types
import unittest


MDOC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MDOC_ROOT not in sys.path:
    sys.path.insert(0, MDOC_ROOT)

sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)))
sys.modules.setdefault("PIL", types.SimpleNamespace(Image=types.SimpleNamespace()))
sys.modules.setdefault("PIL.Image", types.SimpleNamespace())
sys.modules.setdefault("pymupdf", types.SimpleNamespace())


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        values = [row[key] for row in self.rows]
        return types.SimpleNamespace(mean=lambda: sum(values) / len(values))


sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=FakeDataFrame))


class FakeCompiledGraph:
    def __init__(self, graph):
        self.graph = graph
        self.edges = graph.edges

    def invoke(self, state):
        state = dict(state)
        completed = set()

        while True:
            progressed = False
            for name, fn in self.graph.nodes.items():
                if name in completed:
                    continue
                parents = self.graph.parents.get(name, set())
                if not parents.issubset(completed | {"__start__"}):
                    continue
                if "__start__" in parents or parents:
                    state.update(fn(state))
                    completed.add(name)
                    progressed = True
            if "__end__" in self.graph.edges.get("sum", []) and "sum" in completed:
                return state
            if not progressed:
                return state
            state.update(self.graph.nodes[current](state))


class FakeStateGraph:
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = {}
        self.parents = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, start, end):
        self.edges.setdefault(start, []).append(end)
        self.parents.setdefault(end, set()).add(start)

    def compile(self):
        return FakeCompiledGraph(self)


fake_langgraph = types.ModuleType("langgraph")
fake_graph = types.ModuleType("langgraph.graph")
fake_graph.StateGraph = FakeStateGraph
fake_graph.START = "__start__"
fake_graph.END = "__end__"
sys.modules["langgraph"] = fake_langgraph
sys.modules["langgraph.graph"] = fake_graph

from agents.mdoc_graph import build_mdoc_graph


class FakeAgent:
    def __init__(self, name, response, reflection=None):
        self.name = name
        self.response = response
        self.reflection = reflection
        self.calls = []
        self.config = types.SimpleNamespace(
            agent=types.SimpleNamespace(critical_prompt="reflect")
        )

    def predict(self, question, texts=None, images=None, with_sys_prompt=True):
        self.calls.append(
            {
                "question": question,
                "texts": texts,
                "images": images,
                "with_sys_prompt": with_sys_prompt,
            }
        )
        return self.response, []

    def self_reflect(self, prompt=None, add_to_message=True):
        return self.reflection


class FakeSystem:
    def __init__(self):
        self.image_agent = FakeAgent("image", "image answer")
        self.text_agent = FakeAgent("text", "text answer")
        self.general_agent = FakeAgent(
            "general",
            "general answer",
            '{"text": "text clue", "image": "image clue"}',
        )
        self.agents = [self.image_agent, self.text_agent, self.general_agent]
        self.sum_input = None

    def sum(self, content):
        self.sum_input = content
        return "final answer", []


class MDocGraphTests(unittest.TestCase):
    def test_graph_runs_mdoc_agent_chain_and_returns_final_answer(self):
        system = FakeSystem()
        graph = build_mdoc_graph(system)

        result = graph.invoke(
            {
                "question": "What is in the document?",
                "texts": ["text evidence"],
                "images": ["image.png"],
            }
        )

        self.assertEqual(result["final_answer"], "final answer")
        self.assertEqual(result["agent_details"]["General Agent"], "general answer")
        self.assertEqual(result["agent_details"]["Text Agent"], "text answer")
        self.assertEqual(result["agent_details"]["Image Agent"], "image answer")
        self.assertEqual(result["agent_details"]["Sum Agent"], "final answer")
        self.assertIn("text clue", system.text_agent.calls[0]["question"])
        self.assertIn("image clue", system.image_agent.calls[0]["question"])
        self.assertEqual(system.text_agent.calls[0]["images"], None)
        self.assertEqual(system.image_agent.calls[0]["texts"], None)

    def test_graph_branches_text_and_image_after_reflection(self):
        system = FakeSystem()
        graph = build_mdoc_graph(system)

        self.assertEqual(set(graph.edges["reflect"]), {"text", "image"})
        self.assertIn("sum", graph.edges["text"])
        self.assertIn("sum", graph.edges["image"])


class MDocAgentIntegrationTests(unittest.TestCase):
    def test_mdoc_agent_predict_uses_compiled_graph_when_enabled(self):
        import agents.mdoc_agent as mdoc_agent_module

        agent = object.__new__(mdoc_agent_module.MDocAgent)
        agent.use_langgraph = True
        agent.graph = types.SimpleNamespace(
            invoke=lambda state: {
                "final_answer": "graph answer",
                "agent_details": {"Sum Agent": "graph answer"},
            }
        )

        answer, details = agent.predict("q", ["t"], ["i"])

        self.assertEqual(answer, "graph answer")
        self.assertEqual(details["Sum Agent"], "graph answer")

    def test_mdoc_agent_predict_passes_langsmith_config_to_graph(self):
        import agents.mdoc_agent as mdoc_agent_module

        calls = {}

        def invoke(state, config=None):
            calls["config"] = config
            return {
                "final_answer": "graph answer",
                "agent_details": {"Sum Agent": "graph answer"},
            }

        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGSMITH_PROJECT", None)
        agent = object.__new__(mdoc_agent_module.MDocAgent)
        agent.config = types.SimpleNamespace(
            tracing=types.SimpleNamespace(
                enabled=True,
                project="mhier-agent",
                run_name="mdoc-graph",
                tags=["graph"],
                metadata={"run": "test"},
            )
        )
        agent.use_langgraph = True
        agent.graph = types.SimpleNamespace(invoke=invoke)

        agent.predict("q", ["t"], ["i"])

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(os.environ["LANGSMITH_PROJECT"], "mhier-agent")
        self.assertEqual(calls["config"]["run_name"], "mdoc-graph")
        self.assertEqual(calls["config"]["tags"], ["graph"])
        self.assertEqual(calls["config"]["metadata"], {"run": "test"})


if __name__ == "__main__":
    unittest.main()
