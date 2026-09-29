import os
import sys
from pathlib import Path
from dotenv import load_dotenv

mdoc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if mdoc_root not in sys.path:
    sys.path.append(mdoc_root)
mmrag_root = os.path.abspath(os.path.join(mdoc_root, '..', 'MMRAG-DocQA'))
if mmrag_root not in sys.path:
    sys.path.append(mmrag_root)

load_dotenv(Path(mdoc_root).resolve().parents[0] / ".env")
from mydatasets.base_dataset import BaseDataset
from agents.base_agent import Agent
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    cfg.eval_agent.agent = hydra.compose(config_name="agent/"+cfg.eval_agent.agent, overrides=[]).agent
    cfg.eval_agent.model = hydra.compose(config_name="model/"+cfg.eval_agent.model, overrides=[]).model
    dataset = BaseDataset(cfg.dataset)
    eval_agent = Agent(cfg.eval_agent)
    eval_agent.eval_dataset(dataset)
    
if __name__ == "__main__":
    main()
