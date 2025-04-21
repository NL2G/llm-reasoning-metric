from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks
import sys
import json
import re
import pandas as pd
import numpy as np
import argparse as ap
from vllm import LLM, SamplingParams
from rich import print

DEFAULT_INSTRUCTION = "Translate this text from {source_language} to {target_language}, maintaining the tone, accuracy and ensuring fluency: {source_text}"

JUDGE_PROMPT_THINKING = """Please act as an impartial judge and evaluate the quality of the translations provided by two AI assistants in response to the user's request below.
Select the assistant that best adheres to the user's instructions while producing the highest-quality translation overall.
Begin by comparing the two translations and reason before you answer.
Avoid personal opinions or biases, and do not favor one assistant over the other.
Your judgment should be based solely on the quality of the translations and their alignment with the user's instructions.
Be objective and impartial. If both translations are equally good, you can choose the one that you prefer.

Reply strictly in this format: 

<think>
... your reasoning process ...
</think>
<answer>
[A] if Assistant A is better, [B] if Assistant B is better
</answer>

[User Instruction]
{instruction}
[End of User Instruction]

[Start of Assistant A's Response]
{assistant_a_response}
[End of Assistant A's Response]

[Start of Assistant B's Response]
{assistant_b_response}
[End of Assistant B's Response]
"""

JUDGE_PROMPT_NO_THINKING = """Please act as an impartial judge and evaluate the quality of the translations provided by two AI assistants in response to the user's request below. Select the assistant that produces the highest-quality translation overall. Begin by comparing the two translations and provide a verdict. Avoid personal opinions or biases, and do not favor one assistant over the other. Your judgment should be based solely on the quality of the translations and their alignment with the user's instructions. Be objective and impartial. If both translations are equally good, you can choose the one that you prefer.

Deliver your response strictly in this format, do not include any other text: 

Chosen: <[A] if Assistant A is better, [B] if Assistant B is better>

[User Instruction]
{instruction}
[End of User Instruction]

[Start of Assistant A's Response]
{assistant_a_response}
[End of Assistant A's Response]

[Start of Assistant B's Response]
{assistant_b_response}
[End of Assistant B's Response]
"""

def extract_answer_no_thinking(response: str) -> str:
    pattern = r"Chosen:\s*(.{1,15})"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None
    
def extract_answer_thinking(response: str) -> str:
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer:
        return answer.group(1).strip().replace("[", "").replace("]", "")
    else:
        return None

LANG_CODES = {
    'en': 'English',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'cs': 'Czech',
    'uk': 'Ukrainian',
    'de': 'German',
}

PROMPTS = {
    'thinking': JUDGE_PROMPT_THINKING,
    'no_thinking': JUDGE_PROMPT_NO_THINKING,
}

EXTRACT_ANSWER = {
    'thinking': extract_answer_thinking,
    'no_thinking': extract_answer_no_thinking,
}

def make_scores(
    sources: list[str],
    system_outputs: dict[str, list[str]],
    source_lang: str,
    target_lang: str,
    prompt_template: str,
    use_system_prompt: bool = True,
    vllm_client: LLM = None
) -> dict[str, list[float]]:
    """
    Make scores for a given set of sources and system outputs.
    """
    request_batch = []
    meta_data = []
    for source_idx, source in enumerate(sources):
        systems = list(system_outputs.keys())
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                instruction = DEFAULT_INSTRUCTION.format(
                    source_language=source_lang,
                    target_language=target_lang,
                    source_text=source
                )
                prompt = prompt_template.format(
                    instruction=instruction,
                    assistant_a_response=system_outputs[systems[i]][source_idx],
                    assistant_b_response=system_outputs[systems[j]][source_idx]
                )
                messages = []
                if use_system_prompt:
                    messages.append({
                        "role": "system", 
                        "content": 'You are a helpful translation evaluator. You will provide a verdict in a strict format, do not include any other text. Just word "Chosen: A" or "Chosen: B".'
                    })

                messages.append({"role": "user", "content": prompt})
                request_batch.append(messages)
                meta_data.append({
                    "source_idx": source_idx,
                    "system_a": systems[i],
                    "system_b": systems[j]
                })
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=2048
    )
    results = vllm_client.chat(
        messages=request_batch[:4000], 
        sampling_params=sampling_params,
        add_generation_prompt=True,
        use_tqdm=True
    )
    results = [result.outputs[0].text for result in results]
    # Process responses

    # Initialize win counts for each system for each source
    system_wins = {system: [0] * len(sources) for system in system_outputs.keys()}
    system_comparisons = {system: [0] * len(sources) for system in system_outputs.keys()}

    for response, meta in zip(results, meta_data):
        source_idx = meta["source_idx"]
        system_a = meta["system_a"]
        system_b = meta["system_b"]
        
        try:
        # Parse the response - look for 'A' or 'B' in the first few characters of the response
            choice = EXTRACT_ANSWER[prompt_type](response)
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

    return scores


def evaluate(
    model_name: str,
    prompt_type: str,
    language_pairs: list[str] = ["ja-zh"],
    use_system_prompt: bool = True,
    competition: str = "wmt24",
    output: str = "metric.json"
):
    
    vllm_client = LLM(
        model=model_name,
        enable_reasoning=True,
        reasoning_parser="deepseek_r1",
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.9,
        compilation_config=3,
        max_model_len=4096,
    )

    evs_dict = {
        (competition, lp): data.EvalSet(competition, lp, True) for lp in language_pairs
    }

    metric_name = output.split(".")[0]

    for lp in language_pairs:
        evs = evs_dict[(competition, lp)]
        for refname, ref in evs.all_refs.items():
            seg_scores = make_scores(
                evs.src,
                evs.sys_outputs,
                evs.lp.split('-')[0],
                evs.lp.split('-')[1],
                prompt_template=PROMPTS[prompt_type],
                use_system_prompt=use_system_prompt,
                vllm_client=vllm_client
            )
            sys_scores = {}
            for system, scores in seg_scores.items():
                sys_scores[system] = [np.mean(scores)]
            evs.AddMetric(metric_name, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {refname}, 'seg', seg_scores, replace=True)

    # Add new metric to the primary lists, so it will get picked up when tasks get
    # run with primary=True (avoiding having to evaluate all contrastive
    # submissions as well).

    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(evs.primary_metrics | {metric_name})

    wmt_tasks, wts = tasks.WMT24(language_pairs, k=0)

    # Takes about 3 minutes.
    new_results = wmt_tasks.Run(eval_set_dict=evs_dict)

    metric_df = {
        'name': metric_name,
    }
    for result in new_results:
        attr_vals = result.attr_vals
        corr_ranks = result.corr_ranks
        metric_df[f"{attr_vals['lang']} / {attr_vals['level']} / {attr_vals['corr_fcn']}"] = corr_ranks[metric_name][0]

    print(metric_df)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metric_df, f, indent=4, ensure_ascii=False)

    avg_corrs = new_results.AverageCorrs(wts)

    table = new_results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header='avg-corr',
        attr_list=['lang', 'level', 'corr_fcn'],
        nicknames={'KendallWithTiesOpt': 'acc-t'},
        fmt='text',
        baselines_metainfo=meta_info.WMT24)

    return table


def main():
    parser = ap.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["thinking", "no_thinking"])
    parser.add_argument("--language_pairs", type=str, required=True)
    parser.add_argument("--use_system_prompt", default=False, action="store_true")
    parser.add_argument("--competition", type=str, required=True, choices=["wmt24"])
    parser.add_argument("--output", type=str, default="metric.json")
    args = parser.parse_args()

    print(f"Running with args: {args}")

    language_pairs = args.language_pairs.split(" ")

    table = evaluate(
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        language_pairs=language_pairs,
        use_system_prompt=args.use_system_prompt,
        competition=args.competition,
        output=args.output
    )

    print(table)

if __name__ == "__main__":
    main()