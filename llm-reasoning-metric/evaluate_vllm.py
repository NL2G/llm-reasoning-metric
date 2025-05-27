from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    logger.info(f"Evaluating config: {OmegaConf.to_yaml(cfg)}")

    
