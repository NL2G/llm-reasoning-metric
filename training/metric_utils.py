import re
from sympy.parsing.latex import parse_latex
import json
from ifeval_functions import IF_FUNCTIONS_MAP


DEFAULT_INSTRUCTION = "Translate this text from {source_language} to {target_language}, maintaining the tone, accuracy and ensuring fluency: {source_text}"

JUDGE_PROMPT_THINKING = """Please act as an impartial judge and evaluate the quality of the translations provided by two AI assistants in response to the user's request below.
Select the assistant that best adheres to the user's instructions while producing the highest-quality translation overall.
Begin by comparing the two translations and reason before you answer.
Avoid personal opinions or biases, and do not favor one assistant over the other.
Your judgment should be based solely on the quality of the translations and their alignment with the user's instructions.
Be objective and impartial. If both translations are equally good, you can choose the one that you prefer.

After providing your explanation, response strictly in this format: 

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

<[A] if Assistant A is better, [B] if Assistant B is better>

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

SYSTEM_NO_CHOSEN = 'You are a helpful translation evaluator. You will provide a verdict in a strict format, do not include any other text. Just letter "A" or "B".'
SYSTEM_CHOSEN = 'You are a helpful translation evaluator. You will provide a verdict in a strict format, do not include any other text. Just words "Chosen: A" or "Chosen: B".'


LANG_CODES = {
    'en': 'English',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'cs': 'Czech',
    'uk': 'Ukrainian',
    'de': 'German',
    'es': 'Spanish'
}

def pairwise_ranking_grpo_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        lang1, lang2 = example['lp'].split('-')
        source_text = example['src']
        instruction = DEFAULT_INSTRUCTION.format(
            source_language=LANG_CODES[lang1],
            target_language=LANG_CODES[lang2],
            source_text=source_text
        )

        input_message = JUDGE_PROMPT_THINKING.format(
            instruction=instruction,
            assistant_a_response=example["hyp0"],
            assistant_b_response=example["hyp1"]
        )

        answer = "A" if example["best_hyp"] == 0 else "B"

        return {
            "prompt": [
                {"role": "user", "content": input_message},
            ],
            "answer": answer,
            "dataset": "mt-ranking"
        }

    return transform_fn, {
        'remove_columns': [
            "lp", "src", "hyp0", "hyp1", "best_hyp",
            "ref", 'score0', 'score1', 'score_diff',
            "score_name", "system0", "system1"
        ],
    }


def pairwise_ranking_for_sft(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        lang1, lang2 = example['lp'].split('-')
        source_text = example['src']
        instruction = DEFAULT_INSTRUCTION.format(
            source_language=LANG_CODES[lang1],
            target_language=LANG_CODES[lang2],
            source_text=source_text
        )

        input_message = JUDGE_PROMPT.format(
            instruction=instruction,
            assistant_a_response=example["hyp0"],
            assistant_b_response=example["hyp1"]
        )

        answer = "A" if example["best_hyp"] == 0 else "B"
        answer = f"Chosen: {answer}"

        return {
            "messages": [
                {"role": "system", "content": "You are a helpful translation evaluator. You will provide a verdict in a strict format, do not include any other text. Just word 'Chosen: A' or 'Chosen: B'."},
                {"role": "user", "content": input_message},
                {"role": "assistant", "content": answer}
            ]
        }

    return transform_fn, {
        'remove_columns': [
            "lp", "src", "hyp0", "hyp1", "best_hyp",
            "ref", 'score0', 'score1', 'score_diff',
            "score_name", "system0", "system1"
        ],
    }


def tulu_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": example["messages"],
            "answer": example["ground_truth"],
            "dataset": example["dataset"]
        }

    return transform_fn, {
        'remove_columns': ['messages', 'ground_truth', 'constraint_type', 'constraint']
    }


def extract_answer(response):
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer:
        answer = answer.group(1).strip()
    else:
        answer = "NOTHING"
    # remove [] if present
    return answer.replace('[', '').replace(']', '')


def extract_thinking(response):
    thinking = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if thinking:
        thinking = thinking.group(1).strip()
    else:
        thinking = "NOTHING"
    return thinking


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
    if text.count("</answer>") == 1:
        count += 0.125
        # penalize extra tokens after the answer tag
        count -= (len(text.split("</answer>")[-1]) - 1)*0.001
    return count


def string_exact_match(extracted_answer, ground_truth):
    return extracted_answer == ground_truth

def number_exact_match(extracted_answer, ground_truth):
    try:
        extracted_answer = float(extracted_answer.replace(",", "."))
        ground_truth = float(ground_truth.replace(",", "."))
        return extracted_answer == ground_truth
    except ValueError:
        return False
    
def latex_exact_match(extracted_answer, ground_truth):
    try:
        extracted_answer = parse_latex(extracted_answer).simplify()
        ground_truth = parse_latex(ground_truth).simplify()
        return extracted_answer.equals(ground_truth)
    except Exception:
        return False
    

def ifeval_validator(extracted_answer, ground_truth):
    if isinstance(ground_truth, str):
        constraint = json.loads(ground_truth)
    else:
        constraint = ground_truth

    func_name = constraint.pop("func_name")
    func = IF_FUNCTIONS_MAP[func_name]
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    func_result = func(extracted_answer, **non_none_args)
    return func_result

REWARD_FUNCTIONS = {
    "mt-ranking": string_exact_match,
    "gsm8k": number_exact_match,
    "MATH": latex_exact_match,
    "ifeval": ifeval_validator
}

def reward_answer_correctness(completions, answer, dataset, **kwargs):
    pred_answers = [extract_answer(completion[0]['content']) for completion in completions]
    rewards = []
    for pred_answer, true_answer, item_dataset in zip(pred_answers, answer, dataset):
        if item_dataset not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown dataset: {item_dataset}")
        reward = 2.0 if REWARD_FUNCTIONS[item_dataset](pred_answer, true_answer) else 0.0
        rewards.append(reward)
    return rewards


def reward_format_correctness(completions, **kwargs):
    return [
        count_xml(completion[0]['content']) for completion in completions
    ]

