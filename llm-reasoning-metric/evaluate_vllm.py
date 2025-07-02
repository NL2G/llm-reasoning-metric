from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks
import json
import pandas as pd
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from rich.logging import RichHandler
from verl_reward import extract_answer, extract_letter, extract_answer_no_tag
from prompts import USER_PROMPTS, SYSTEM_PROMPTS, LANG_CODES, parse_and_check_numerical_answer, GEMBA_FEW_SHOTS

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


COMPETITIONS = {
    "wmt24": tasks.WMT24,
    "wmt23": tasks.WMT23,
}

META_INFO = {
    "wmt24": meta_info.WMT24,
    "wmt23": meta_info.WMT23,
}


def mt_ranking_eval(
    sources: list[str],
    system_outputs: dict[str, list[str]],
    source_lang: str,
    target_lang: str,
    system_prompt: str,
    user_prompt_template: str,
    llm: LLM = None,
    reasoning_effort: str | None = None,
    sampling_params: dict[str, any] = None,
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """
    Make scores for a given set of sources and system outputs.
    """
    request_batch = []
    prompts = []
    meta_data = []
    source_lang_name = LANG_CODES[source_lang]
    target_lang_name = LANG_CODES[target_lang]

    if reasoning_effort is not None:
        system_prompt = "<reasoning_effort>" + reasoning_effort + "</reasoning_effort>\n" + system_prompt
    else:
        system_prompt = system_prompt

    for source_idx, source in enumerate(sources):
        systems = list(system_outputs.keys())
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                messages = []
                messages.append({
                    "role": "system", 
                    "content": system_prompt
                })
                messages.append({"role": "user", "content": user_prompt_template.format(
                    source_language=source_lang_name,
                    target_language=target_lang_name,
                    source_text=source,
                    assistant_a_response=system_outputs[systems[i]][source_idx],
                    assistant_b_response=system_outputs[systems[j]][source_idx]
                )})

                request_batch.append(messages)
                prompts.append(messages[1]["content"])

                meta_data.append({
                    "source_idx": source_idx,
                    "system_a": systems[i],
                    "system_b": systems[j]
                })

    logger.info(f"=> Running MT-ranking for {len(request_batch)} prompts")
    logger.info(f"=> Reasoning effort: {reasoning_effort}")
    logger.info(f"=> First prompt: {request_batch[0]}")

    results = llm.chat(
        request_batch, sampling_params=SamplingParams(**sampling_params),
        add_generation_prompt=True, use_tqdm=True,
    )
    results = [result.outputs[0].text for result in results]
    

    traces_df = pd.DataFrame({
        "prompt": prompts,
        "result": results
    })

    # Initialize win counts for each system for each source
    system_wins = {system: [0] * len(sources) for system in system_outputs.keys()}
    system_comparisons = {system: [0] * len(sources) for system in system_outputs.keys()}

    for response, meta in zip(results, meta_data):
        source_idx = meta["source_idx"]
        system_a = meta["system_a"]
        system_b = meta["system_b"]
        
        try:
        # Parse the response - look for 'A' or 'B' in the first few characters of the response
            parsed_answer = extract_answer(response)
            choice, _ = extract_letter(parsed_answer)
        except Exception as e:
            choice = None

        if choice is not None:
            if "A" in choice:  # Check beginning of response
                # System A wins
                system_wins[system_a][source_idx] += 1
            elif "B" in choice:
                # System B wins
                system_wins[system_b][source_idx] += 1
            
            # Update comparison counts
            system_comparisons[system_a][source_idx] += 1
            system_comparisons[system_b][source_idx] += 1
    
    # Calculate scores (normalized by number of comparisons)
    scores = {}
    for system in system_outputs.keys():
        scores[system] = [
            wins / comps if comps > 0 else 0.0 
            for wins, comps in zip(system_wins[system], system_comparisons[system])
        ]

    return scores, traces_df


def gemba_da_eval(
    sources: list[str],
    system_outputs: dict[str, list[str]],
    source_lang: str,
    target_lang: str,
    system_prompt: str,
    user_prompt_template: str,
    llm: LLM = None,
    reasoning_effort: str | None = None,
    sampling_params: dict[str, any] = None,
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """
    Make scores for a given set of sources and system outputs.
    """
    request_batch = []
    prompts = []
    meta_data = []
    source_lang_name = LANG_CODES[source_lang]
    target_lang_name = LANG_CODES[target_lang]

    if reasoning_effort is not None:
        system_prompt = "<reasoning_effort>" + reasoning_effort + "</reasoning_effort>\n" + system_prompt
    else:
        system_prompt = system_prompt

    for source_idx, source in enumerate(sources):
        systems = list(system_outputs.keys())
        for i in range(len(systems)):
            messages = []
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
            messages.append({"role": "user", "content": user_prompt_template.format(
                source_language=source_lang_name,
                target_language=target_lang_name,
                source_text=source,
                translation=system_outputs[systems[i]][source_idx]
            )})

            request_batch.append(messages)
            prompts.append(messages[1]["content"])
            meta_data.append({
                "source_idx": source_idx,
                "system": systems[i]
            })

    logger.info(f"=> Running DA-like for {len(request_batch)} prompts")
    logger.info(f"=> Reasoning effort: {reasoning_effort}")
    logger.info(f"=> First prompt: {request_batch[0]}")

    results = llm.chat(
        request_batch, sampling_params=SamplingParams(**sampling_params),
        add_generation_prompt=True, use_tqdm=True,
    )
    results = [result.outputs[0].text for result in results]
    

    traces_df = pd.DataFrame({
        "prompt": prompts,
        "result": results
    })

    scores = {k: [None]*len(sources) for k in system_outputs.keys()}
    for result, meta in zip(results, meta_data):
        source_idx = meta["source_idx"]
        system = meta["system"]
        try:
            score_value = extract_answer(result)
            score_value = score_value.replace("*", "")
            score_value = parse_and_check_numerical_answer(score_value, min=0, max=100)
        except Exception as e:
            score_value = None

        if score_value is None:
            score_value = 0.0

        scores[system][source_idx] = score_value

    return scores, traces_df


def gemba_esa_eval(
    sources: list[str],
    system_outputs: dict[str, list[str]],
    source_lang: str,
    target_lang: str,
    system_prompt: str,
    user_spans_template: str,
    user_ranking_template: str,
    llm: LLM = None,
    reasoning_effort: str | None = None,
    sampling_params: dict[str, any] = None,
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """
    Make scores for a given set of sources and system outputs.
    """
    request_batch = []
    prompts = []
    meta_data = []
    source_lang_name = LANG_CODES[source_lang]
    target_lang_name = LANG_CODES[target_lang]

    if reasoning_effort is not None:
        system_prompt = "<reasoning_effort>" + reasoning_effort + "</reasoning_effort>\n" + system_prompt
    else:
        system_prompt = system_prompt

    for source_idx, source in enumerate(sources):
        systems = list(system_outputs.keys())
        for i in range(len(systems)):
            messages = []
            messages.append({
                "role": "system", 
                "content": system_prompt
            })

            for shot in GEMBA_FEW_SHOTS:
                messages.append({"role": "user", "content": user_spans_template.format(
                    source_language=source_lang_name,
                    target_language=target_lang_name,
                    source_text=shot["source_text"],
                    translation=shot["translation"]
                )})
                messages.append({"role": "assistant", "content": shot["response"]})

            messages.append({"role": "user", "content": user_spans_template.format(
                source_language=source_lang_name,
                target_language=target_lang_name,
                source_text=source,
                translation=system_outputs[systems[i]][source_idx]
            )})

            request_batch.append(messages)
            meta_data.append({
                "source_idx": source_idx,
                "system": systems[i],
                "source_text": source,
                "translation": system_outputs[systems[i]][source_idx],
            })

    logger.info(f"=> Running error spans extraction for {len(request_batch)} prompts")
    logger.info(f"=> Reasoning effort: {reasoning_effort}")
    logger.info(f"=> First prompt: {request_batch[0]}")
    results = llm.chat(
        request_batch, sampling_params=SamplingParams(**sampling_params),
        add_generation_prompt=True, use_tqdm=True,
    )
    spans_results = [extract_answer_no_tag(result.outputs[0].text) for result in results]
    ranking_batch = []
    for span, meta in zip(spans_results, meta_data):
        messages = []
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        messages.append({"role": "user", "content": user_ranking_template.format(
            source_language=source_lang_name,
            target_language=target_lang_name,
            source_text=meta["source_text"],
            translation=meta["translation"],
            error_spans=span
        )})
        ranking_batch.append(messages)
        prompts.append(messages[-1]["content"])

    logger.info(f"=> Running ranking for {len(ranking_batch)} prompts")
    ranking_results = llm.chat(
        ranking_batch, sampling_params=SamplingParams(**sampling_params),
        add_generation_prompt=True, use_tqdm=True,
    )
    ranking_results = [extract_answer_no_tag(result.outputs[0].text) for result in ranking_results]

    traces_df = pd.DataFrame({
        "prompt": prompts,
        "result": ranking_results
    })

    scores = {k: [None]*len(sources) for k in system_outputs.keys()}
    for result, meta in zip(ranking_results, meta_data):
        source_idx = meta["source_idx"]
        system = meta["system"]
        try:
            score_value = parse_and_check_numerical_answer(result, min=0, max=100)
        except Exception as e:
            score_value = None

        if score_value is None:
            score_value = 0.0

        scores[system][source_idx] = score_value

    return scores, traces_df


@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    logger.info(f"Evaluating config: {OmegaConf.to_yaml(cfg)}")

    experiment = cfg.experiment

    logger.info(f"Evaluating model: {experiment.model}")
    reasoning_effort = experiment.get("reasoning_effort", None)
    if reasoning_effort is not None:
        output_model_path = f"{experiment.model_id}#{reasoning_effort}@[{experiment.kind}]"
    else:
        output_model_path = f"{experiment.model_id}@[{experiment.kind}]"

    logger.info(f"Output model path: {output_model_path}")
    result_dir = Path(cfg.outputs_dir) / output_model_path
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Spinning up vLLM")
    llm = LLM(
        model=experiment.model,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=experiment.sampling_params.max_tokens,
        quantization="fp8",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        compilation_config=CompilationConfig(
            level=3
        )
    )

    evs_dict = {
        (cfg.competition, lp): data.EvalSet(cfg.competition, lp, True) for lp in cfg.language_pairs
    }

    
    if reasoning_effort is None:
        metric_name = experiment.model + "@[" + experiment.kind + "]"
    else:
        metric_name = experiment.model + "#" + reasoning_effort + "@[" + experiment.kind + "]"

    for lp in cfg.language_pairs:
        evs = evs_dict[(cfg.competition, lp)]
        lp_dir = result_dir / lp
        lp_dir.mkdir(parents=True, exist_ok=True)

        if experiment.kind == "mt-ranking":
            seg_scores, traces_df = mt_ranking_eval(
                evs.src,
                evs.sys_outputs,
                evs.lp.split('-')[0],
                evs.lp.split('-')[1],
                system_prompt=SYSTEM_PROMPTS[experiment.kind],
                user_prompt_template=USER_PROMPTS[experiment.kind],
                llm=llm,
                reasoning_effort=reasoning_effort,
                sampling_params=experiment.sampling_params
            )
        elif experiment.kind == "gemba-da-like":
            seg_scores, traces_df = gemba_da_eval(
                evs.src,
                evs.sys_outputs,
                evs.lp.split('-')[0],
                evs.lp.split('-')[1],
                system_prompt=SYSTEM_PROMPTS[experiment.kind],
                user_prompt_template=USER_PROMPTS[experiment.kind],
                llm=llm,
                reasoning_effort=reasoning_effort,
                sampling_params=experiment.sampling_params
            )
        elif experiment.kind == "gemba-esa":
            seg_scores, traces_df = gemba_esa_eval(
                evs.src,
                evs.sys_outputs,
                evs.lp.split('-')[0],
                evs.lp.split('-')[1],
                system_prompt=SYSTEM_PROMPTS[experiment.kind],
                user_spans_template=USER_PROMPTS[experiment.kind + "-error-spans"],
                user_ranking_template=USER_PROMPTS[experiment.kind + "-ranking"],
                llm=llm,
                reasoning_effort=reasoning_effort,
                sampling_params=experiment.sampling_params
            )
        else:
            raise ValueError(f"Unknown model kind: {experiment.kind}")
            
        traces_df['lp'] = lp
        sys_scores = {}
        for system, scores in seg_scores.items():
            sys_scores[system] = [np.mean(scores)]

        for ref in evs.all_refs.keys():    
            evs.AddMetric(metric_name, {ref}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {ref}, 'seg', seg_scores, replace=True)
        
        traces_df = traces_df.sample(n=cfg.traces_sample_size).reset_index(drop=True)
        traces_df.to_csv(str(lp_dir / f'traces.csv'), index=False)
            
    
        for evs in evs_dict.values():
            evs.SetPrimaryMetrics(evs.primary_metrics | {metric_name})
        
        wmt_tasks, wts = COMPETITIONS[cfg.competition](cfg.language_pairs, k=0)
        new_results = wmt_tasks.Run(eval_set_dict=evs_dict)

        metric_df = {
            "name": metric_name,
        }
        for result in new_results:
            attr_vals = result.attr_vals
            corr_ranks = result.corr_ranks
            print(corr_ranks)
            metric_df[f"{attr_vals['lang']} / {attr_vals['level']} / {attr_vals['corr_fcn']}"] = corr_ranks[metric_name][0]

        logger.info(f"Metric results: {metric_df}")
        logger.info(f"Saving metric results to {lp_dir / 'metrics.json'}")
        with open(lp_dir / 'metrics.json', "w", encoding="utf-8") as f:
            json.dump(metric_df, f, indent=4, ensure_ascii=False)

        avg_corrs = new_results.AverageCorrs(wts)

        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header='avg-corr',
            attr_list=['lang', 'level', 'corr_fcn'],
            nicknames={'KendallWithTiesOpt': 'acc-t'},
            fmt='text',
            baselines_metainfo=META_INFO[cfg.competition])
        
        logger.info(f"Saving table to {lp_dir / 'table.txt'}")
        with open(lp_dir / 'table.txt', "w", encoding="utf-8") as f:
            f.write(str(table))
        

if __name__ == "__main__":
    main()
