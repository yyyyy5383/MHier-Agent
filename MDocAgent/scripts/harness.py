import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import hydra


mdoc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if mdoc_root not in sys.path:
    sys.path.append(mdoc_root)
mmrag_root = os.path.abspath(os.path.join(mdoc_root, "..", "MMRAG-DocQA"))
if mmrag_root not in sys.path:
    sys.path.append(mmrag_root)

load_dotenv(Path(mdoc_root).resolve().parents[0] / ".env")

def has_ground_truth(samples, gt_key):
    if not samples:
        return False
    return all(gt_key in sample and sample[gt_key] not in (None, "") for sample in samples)


def write_summary(
    path,
    *,
    run_name,
    result_dir,
    prediction_path=None,
    eval_path=None,
    sample_count=0,
    eval_sample_count=0,
    average_binary_correctness=None,
    eval_skipped=False,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = {
        "run_name": run_name,
        "result_dir": result_dir,
        "prediction_path": prediction_path,
        "eval_path": eval_path,
        "sample_count": sample_count,
        "eval_sample_count": eval_sample_count,
        "average_binary_correctness": average_binary_correctness,
        "eval_skipped": eval_skipped,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return path


def configure_mdoc_agent(cfg):
    from agents.mdoc_agent import MDocAgent

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_config.agent = hydra.compose(config_name="agent/" + agent_name, overrides=[]).agent
        agent_config.model = hydra.compose(config_name="model/" + model_name, overrides=[]).model

    sum_agent = cfg.mdoc_agent.sum_agent
    sum_agent.agent = hydra.compose(config_name="agent/" + sum_agent.agent, overrides=[]).agent
    sum_agent.model = hydra.compose(config_name="model/" + sum_agent.model, overrides=[]).model
    return MDocAgent(cfg.mdoc_agent)


def configure_eval_agent(cfg):
    from agents.base_agent import Agent

    cfg.eval_agent.agent = hydra.compose(config_name="agent/" + cfg.eval_agent.agent, overrides=[]).agent
    cfg.eval_agent.model = hydra.compose(config_name="model/" + cfg.eval_agent.model, overrides=[]).model
    return Agent(cfg.eval_agent)


def run_harness(cfg):
    dataset = None
    prediction_path = None
    sample_count = 0
    eval_summary = None
    eval_skipped = True

    if cfg.harness.run_predict or cfg.harness.run_eval:
        from mydatasets.base_dataset import BaseDataset

        dataset = BaseDataset(cfg.dataset)

    if cfg.harness.run_predict:
        mdoc_agent = configure_mdoc_agent(cfg)
        mdoc_agent.predict_dataset(dataset, resume_path=cfg.harness.resume_path)

    if cfg.harness.run_predict or cfg.harness.run_eval:
        samples, prediction_path = dataset.load_latest_results()
        sample_count = len(samples)

    if cfg.harness.run_eval:
        samples, _ = dataset.load_latest_results()
        if not has_ground_truth(samples, dataset.config.gt_key):
            message = (
                f"latest results are missing ground truth key '{dataset.config.gt_key}'. "
                "Set dataset.gt_key to the correct field or disable harness.run_eval."
            )
            if cfg.harness.require_gt:
                raise ValueError(message)
            print(message)
        else:
            eval_agent = configure_eval_agent(cfg)
            eval_summary = eval_agent.eval_dataset(dataset)
            eval_skipped = False

    summary_path = write_summary(
        str(cfg.harness.summary_path),
        run_name=str(cfg["run-name"]),
        result_dir=str(dataset.config.result_dir if dataset else cfg.dataset.result_dir),
        prediction_path=prediction_path,
        eval_path=eval_summary["results_path"] if eval_summary else None,
        sample_count=sample_count,
        eval_sample_count=eval_summary["sample_count"] if eval_summary else 0,
        average_binary_correctness=(
            eval_summary["average_binary_correctness"] if eval_summary else None
        ),
        eval_skipped=eval_skipped,
    )
    print(f"Save harness summary to {summary_path}.")
    return summary_path


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    run_harness(cfg)


if __name__ == "__main__":
    main()
