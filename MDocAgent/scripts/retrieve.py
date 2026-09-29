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
from retrieval.base_retrieval import BaseRetrieval
import hydra
import importlib

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.retrieval.cuda_visible_devices
    retrieval_class_path = cfg.retrieval.class_path
    module_name, class_name = retrieval_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    retrieval_class = getattr(module, class_name)
    
    dataset = BaseDataset(cfg.dataset)
    retrieval_model:BaseRetrieval = retrieval_class(cfg.retrieval)
    retrieval_model.find_top_k(dataset)

if __name__ == "__main__":
    main()
