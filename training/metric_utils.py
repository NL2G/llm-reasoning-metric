import re

DEFAULT_INSTRUCTION = "Translate this text from {source_language} to {target_language}, maintaining the tone, accuracy and ensuring fluency: {source_text}"

JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the translations provided by two AI assistants in response to the user's request below.
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

        input_message = JUDGE_PROMPT.format(
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
        }

    return transform_fn, {
        'remove_columns': [
            "lp", "src", "hyp0", "hyp1", "best_hyp",
            "ref", 'score0', 'score1', 'score_diff',
            "score_name", "system0", "system1"
        ],
    }


def extract_answer(response):
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer:
        answer = answer.group(1).strip()
    else:
        answer = "NOTHING"
    # remove [] if present
    return answer.replace('[', '').replace(']', '')


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


def reward_answer_correctness(completions, answer, **kwargs):
    answers = [extract_answer(completion[0]['content']) for completion in completions]
    return [
        2.0 if answer == a else 0.0 for a in answers
    ]


def reward_format_correctness(completions, **kwargs):
    return [
        count_xml(completion[0]['content']) for completion in completions
    ]

def choice_reward(completions, **kwargs):
    return [
        0.5 if extract_answer(completion[0]['content']) in {'A', 'B'} else 0.0
        for completion in completions
    ]
