import datasets as ds
from prompts import SYSTEM_PROMPTS, USER_PROMPTS, LANG_CODES
import argparse as ap
import hydra
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf
import logging
import json
import random
from typing import Callable

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def make_reasoning_effort_prefix() -> tuple[str, str]:
    effort = random.choice(["low", "medium", "high", "none"])
    return f"<reasoning_effort>{effort}</reasoning_effort>", effort


def make_math_map_fn(problem_column: str, answer_column: str, add_reasoning_effort: bool = False) -> Callable[[dict, int], dict]:
    
    def verl_format(item: dict, index: int):

        prefix, effort = "", "none"
        if add_reasoning_effort:
            prefix, effort = make_reasoning_effort_prefix()

        system_prompt = SYSTEM_PROMPTS["math"]
        if add_reasoning_effort:
            system_prompt = prefix + "\n" + system_prompt

        if "boxed" not in item[answer_column]:
            answer = "\\boxed{" + item[answer_column] + "}"
        else:
            answer = item[answer_column]

        return {
            "data_source": "math",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPTS["math"].format(problem=item[problem_column])}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "index": index,
                "reasoning_effort": effort
            }
        }
    
    return verl_format


def make_mt_ranking_map_fn(add_reasoning_effort: bool = False) -> Callable[[dict, int], dict]:
    
    def verl_format(item: dict, index: int):
        prefix, effort = "", "none"
        if add_reasoning_effort:
            prefix, effort = make_reasoning_effort_prefix()

        system_prompt = SYSTEM_PROMPTS["mt-ranking"]
        if add_reasoning_effort:
            system_prompt = prefix + "\n" + system_prompt

        lang1, lang2 = item['lp'].split('-')
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPTS["mt-ranking"].format(
                source_language=LANG_CODES[lang1],
                target_language=LANG_CODES[lang2],
                source_text=item['src'],
                assistant_a_response=item['hyp0'],
                assistant_b_response=item['hyp1']
            )}
        ]
        answer = "A" if item["best_hyp"] == 0 else "B"
        return {
            "data_source": 'mt-ranking',
            "prompt": prompt,
            "ability": "translation-eval",
            "reward_model": {
                'style': 'rule',
                'ground_truth': answer,
            },
            "extra_info": {
                "index": index,
                "reasoning_effort": effort
            }
        }
    
    return verl_format


def make_mt_da_map_fn(add_reasoning_effort: bool = False) -> Callable[[dict, int], dict]:
    
    def verl_format(item: dict, index: int):

        prefix, effort = "", "none"
        if add_reasoning_effort:
            prefix, effort = make_reasoning_effort_prefix()

        system_prompt = SYSTEM_PROMPTS["gemba-da-like"]
        if add_reasoning_effort:
            system_prompt = prefix + "\n" + system_prompt

        lang1, lang2 = item['lp'].split('-')
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPTS["gemba-da-like"].format(
                source_language=LANG_CODES[lang1],
                target_language=LANG_CODES[lang2],
                source_text=item['src'],
                translation=item['hyp']
            )}
        ]
        answer = int(min(max(item['score'], 0), 100)) # clip score to 0-100
        return {
            "data_source": 'mt-da',
            "prompt": prompt,
            "ability": "translation-eval",
            "reward_model": {
                'style': 'rule',
                'ground_truth': str(answer),
            },
            "extra_info": {
                "index": index,
                "reasoning_effort": effort
            }
        }
    
    return verl_format


@hydra.main(config_path="configs", config_name="prepare-data")
def main(cfg: DictConfig):

    logger.info(f"Preparing config: {OmegaConf.to_yaml(cfg)}")

    for i, dataset in enumerate(cfg.datasets):
        logger.info(f"Preparing dataset {i+1}/{len(cfg.datasets)}: {dataset.file}")

        data = ds.load_dataset(dataset.source_dataset)['train']
        add_reasoning_effort = dataset.add_reasoning_effort
        logger.info(f"Loaded dataset: {data}")
        
        if dataset.kind == "mt-ranking":
            if dataset.subsets is not None:
                subsets = set(dataset.subsets)
                logger.info(f"Filtering dataset to subsets: {subsets}")
                data = data.filter(lambda x: x['source'] in subsets, num_proc=4)

            map_fn = make_mt_ranking_map_fn(add_reasoning_effort)

        elif dataset.kind == "mt-da":
            if dataset.subsets is not None:
                subsets = set(dataset.subsets)
                logger.info(f"Filtering dataset to subsets: {subsets}")
                data = data.filter(lambda x: x['source'] in subsets, num_proc=4)

            map_fn = make_mt_da_map_fn(add_reasoning_effort)

        elif dataset.kind == "math":
            map_fn = make_math_map_fn(dataset.problem_column, dataset.answer_column, add_reasoning_effort)

        else:
            raise ValueError(f"Unknown dataset kind: {dataset.kind}")
        
        columns = data.column_names
        
        logger.info(f"Mapping dataset to Verl format")
        data = data.map(map_fn, batched=False, num_proc=4, with_indices=True, remove_columns=columns)
        data = data.shuffle(seed=42)
        
        if dataset.max_samples != -1:
            logger.info(f"Selecting {dataset.max_samples} samples")
            data = data.select(range(dataset.max_samples))
        
        logger.info(f"Resulting dataset: {data}")
        logger.info(f"First sample: {data[0]}")
        logger.info(f"Last sample: {data[-1]}")
        logger.info(f"Saving dataset to {dataset.file}")
        data.to_parquet(dataset.file)


if __name__ == "__main__":
    main()

