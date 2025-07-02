import datasets as ds
from prompts import SYSTEM_PROMPTS, USER_PROMPTS, LANG_CODES
import argparse as ap
import hydra
from rich.logging import RichHandler
from omegaconf import DictConfig, OmegaConf
import logging
import json
import numpy as np
import random
from typing import Callable
import langcodes as lc

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


WMTPP_EVAL_PAIRS = set([
    "en-cs_CZ", "en-de_DE",
    "en-es_MX", "en-hi_IN",
    "en-ja_JP", "en-ru_RU", 
    "en-uk_UA", "en-zh_CN",
])

WMTPP_TRAIN_PAIRS = set([
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ", "en-da_DK", "en-de_DE",
    "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR", "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR",
    "en-gu_IN", "en-he_IL", "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
    "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN", "en-mr_IN", "en-nl_NL",
    "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR", "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK",
    "en-sl_SI", "en-sr_RS", "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
    "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW", "en-zu_ZA",
]).difference(WMTPP_EVAL_PAIRS)


def wmtpp_make_doc_level(dataset: ds.Dataset) -> ds.Dataset:
    data = dataset.to_pandas()
    examples = []
    for doc, frame in data.groupby('document_id'):
        frame = frame.sort_values(by="segment_id")
        data_ = frame.to_dict(orient='records')[0]
        source = ""
        target = ""
        for item in frame.to_dict(orient='records'):
            source += " " + item['source']
            target += " " + item['target']

        data_['source'] = source
        data_['target'] = target
        examples.append(data_)
    
    return ds.Dataset.from_list(examples)

def make_reasoning_effort_prefix() -> tuple[str, str]:
    effort = np.random.choice(a=["disabled", "unbounded"], size=1, p=[0.3, 0.7])[0]
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


def make_wmtpp_translation_map_fn(add_reasoning_effort: bool = False) -> Callable[[dict, int], dict]:
    
    def verl_format(item: dict, index: int):
        prefix, effort = "", "none"
        if add_reasoning_effort:
            prefix, effort = make_reasoning_effort_prefix()

        system_prompt = SYSTEM_PROMPTS["translation"]
        if add_reasoning_effort:
            system_prompt = prefix + "\n" + system_prompt

        lang1, lang2 = item['lp'].split('-')
        lang1 = 'English'
        lang2 = lc.get(lang2).language_name()

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPTS["translation"].format(
                source_language=lang1,
                target_language=lang2,
                source_text=item['source']
            )}
        ]

        return {
            "data_source": "translation",
            "prompt": prompt,
            "ability": "translation",
            "reward_model": {
                'style': 'rule',
                'ground_truth': item['target']
            },
            "extra_info": {
                "index": index,
                "reasoning_effort": effort,
                "source_lang": lang1,
                "target_lang": lang2,
                "source_text": item['source'],
            }
        }

    
    return verl_format


@hydra.main(config_path="configs", config_name="prepare-data")
def main(cfg: DictConfig):

    logger.info(f"Preparing config: {OmegaConf.to_yaml(cfg)}")

    for i, dataset in enumerate(cfg.datasets):
        logger.info(f"Preparing dataset {i+1}/{len(cfg.datasets)}: {dataset.file}")

        
        add_reasoning_effort = dataset.add_reasoning_effort
        if dataset.kind != "wmtpp-translation":
            data = ds.load_dataset(dataset.source_dataset)['train']
        else:
            language_pairs = WMTPP_TRAIN_PAIRS if dataset.set == 'train' else WMTPP_EVAL_PAIRS
            data = ds.concatenate_datasets([
                ds.load_dataset(dataset.source_dataset, name=pair)['train'] for pair in language_pairs
            ])
            if dataset.doc_level:
                data = wmtpp_make_doc_level(data)

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

        elif dataset.kind == "wmtpp-translation":

            map_fn = make_wmtpp_translation_map_fn(add_reasoning_effort)

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

