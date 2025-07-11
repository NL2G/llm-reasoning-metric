import re
import math_verify as mv
import os
import logging
import transformers as tr
import sys
sys.path.append("/home/hpc/v106be/v106be28/llm-reasoning-metric/llm-reasoning-metric")
from sacrebleu import BLEU
from prompts import SYSTEM_PROMPTS, USER_PROMPTS, parse_and_check_numerical_answer
from openai import OpenAI
from rich import print

logger = logging.getLogger(__name__)

MT_DA_TOLERANCE_RANGE: int = 20
TOKENIZER = tr.AutoTokenizer.from_pretrained(os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"))

EFFORT_TO_TOKEN_COUNT = {
    "low": (128, 1024),
    "medium": (1024, 4096),
    "high": (4096, 8192),
}

def make_mt_eval_request(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPTS["gemba-esa"]
        },
        {
            "role": "user",
            "content": USER_PROMPTS["gemba-da-like"].format(
                source_text=source_text,
                translation=translation,
                source_language=source_lang,
                target_language=target_lang,
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_NAME"),
            messages=prompt,
            temperature=0.0,
            max_tokens=10
        )
        answer = parse_and_check_numerical_answer(response.choices[0].message.content, min=0, max=100)
        if answer is None:
            return 0
        else:
            return answer
    except Exception as e:
        logger.error(f"Error making MT eval request: {e}")
        return 0


def extract_answer(response) -> str | None:
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer:
        answer = answer.group(1).strip().replace("`", "")
    else:
        answer = None
    return answer


def extract_answer_no_tag(response) -> str | None:
    return response.split("</think>")[-1].strip()


def extract_thinking(response) -> str | None:
    thinking = re.search(r'<think>(.*)</think>', response, re.DOTALL)
    if thinking:
        thinking = thinking.group(1).strip()
    else:
        thinking = None
    return thinking


def count_xml(text: str) -> tuple[int, int]:
    """
    Improved function to:
    - Count presence of required tags <think>, </think>, <answer>, </answer>
    - Count 'extra tokens' (characters) after </answer>
    - Count 'extra tokens' between </think> and <answer>
      (i.e., stuff that's not whitespace or the tags themselves)
    """
    count = 0

    # Tag counts
    if text.count("<think>") == 1:
        count += 1
    if text.count("</think>") == 1:
        count += 1
    if text.count("<answer>") == 1:
        count += 1
    if text.count("</answer>") == 1:
        count += 1

    # Extra tokens after </answer>
    m = re.search(r"</answer>(.*)$", text, re.DOTALL)
    extra_after_answer = 0
    if m:
        extra = m.group(1).strip()
        extra_after_answer = len(extra) if extra else 0

    # Extra tokens between </think> and <answer>
    extra_between = 0
    m = re.search(r"</think>(.*)<answer>", text, re.DOTALL)
    if m:
        between = m.group(1)
        # Remove just whitespace and possible 'noise' newlines
        cleaned = between.strip()
        # Only count as extra if NOT empty
        extra_between = len(cleaned) if cleaned else 0

    extra_tokens = extra_after_answer + extra_between
    return count, extra_tokens


def math_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 1.0
    """
    answer = solution_str.split("</think>")[-1].strip()
    if not answer:
        return 0.0
    
    format_regex = r"\\boxed{.*?}"
    match = re.search(format_regex, answer)
    if not match:
        return 0.0
    
    parsed_answer = mv.parse(answer, extraction_config=[mv.LatexExtractionConfig(),mv.ExprExtractionConfig()])
    ground_truth = mv.parse(ground_truth, extraction_config=[mv.LatexExtractionConfig(),mv.ExprExtractionConfig()])

    verification: bool = mv.verify(ground_truth, parsed_answer)
    
    if verification:
        return 1.0
    else:
        return 0.125 # format correctness reward
    

def mt_da_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 1.0
    """
    parsed_answer = extract_answer(solution_str)
    if not parsed_answer:
        return 0.0
    
    try:
        parsed_answer = int(parsed_answer)
    except ValueError:
        return 0.0
    
    ground_truth = int(ground_truth)
    
    if abs(parsed_answer - ground_truth) <= MT_DA_TOLERANCE_RANGE:
        return 1.0
    else:
        return 0.125 # format correctness reward


def extract_letter(text) -> str | None:
    # Primary pattern: "Chosen: " followed by an uppercase letter
    pattern1 = r'Chosen: ([A-Z])'
    
    match = re.search(pattern1, text, re.DOTALL)
    if match:
        return match.group(1)
    
    return None
    

def mt_ranking_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 1.0
    """
    parsed_answer = extract_answer(solution_str)
    if not parsed_answer:
        return 0.0
    
    answer = extract_letter(parsed_answer)

    if not answer:
        return 0.0 
    
    if answer in {"A", "B"}:
        if answer == ground_truth:
            return 1.0 # full correctness reward
        else:
            return 0.125 # format correctness reward
    else:
        return 0.0
    

def translation_reward_da(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 1.0
    """
    answer = extract_answer(solution_str)
    if not answer:
        return 0.0
    bleu = BLEU(effective_order=True)

    score = bleu.sentence_score(answer, [ground_truth]).score
    
    if score == 0:
        return 0.0
    else:
        return score / 100.0

    
def cumulative_format_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 0.500
    """
    reward = 0.0
    count, extra_tokens = count_xml(solution_str)
    if extra_tokens > 0:
        reward -= extra_tokens * 0.001
    
    reward += count * 0.125
    return reward

def exact_format_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 0.250
    """
    pattern = r'<think>\n(.*?)</think>\n*<answer>\n(.*?)\n</answer>'
    match = re.match(pattern, solution_str, re.DOTALL)
    if not match:
        return 0.0
    else:
        return 0.250
    

def unique_lines_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 0.250
    """
    thinking = extract_thinking(solution_str)
    if thinking is None:
        return 0.0
    
    thinking_lines = thinking.split("\n")
    thinking_lines = [line.strip() for line in thinking_lines if len(line.strip()) > 0]
    total_lines = len(thinking_lines)
    unique_lines = len(set(thinking_lines))
    
    if total_lines == unique_lines:
        return 0.250
    elif abs(total_lines - unique_lines) <= 3:
        return 0.125
    else:
        return 0.0
    

def token_stats(solution_str) -> tuple[int, int]:
    thinking = extract_thinking(solution_str)
    if thinking is None:
        return None, None
    
    thinking = [line.strip() for line in thinking.strip().split("\n") if len(line.strip()) > 0]
    thinking = " ".join(thinking)
    
    
    tokens = TOKENIZER.encode(thinking)
    token_count = len(tokens)
    unique_token_count = len(set(tokens))
    return token_count, unique_token_count
    

def reasoning_effort_reward(solution_str, ground_truth, extra_info=None) -> float:
    """
    Max Reward: 0.500
    """
    effort = extra_info.get("reasoning_effort", "unbounded")
    
    token_count, unique_token_count = token_stats(solution_str)
    if token_count is None:
        return 0.0

    if effort == "disabled":
        if token_count < 10:
            return 0.5
        else:
            return max(0.5 - (token_count - 10) * 0.0025, -0.5)
    
    if effort == "unbounded":
        return 0.5
    
    t1, t2 = EFFORT_TO_TOKEN_COUNT[effort]

    repetition_rate = (token_count - unique_token_count) / token_count
    if repetition_rate > 0.99:
        return 0.0
    
    if repetition_rate < 0.60:
        repetition_rate = 0.0
    
    if t1 <= token_count <= t2:
        return 0.5 - repetition_rate * 0.1
    else:
        return 0.0

TASK_REWARDS = {
    "math": math_reward,
    "mt-ranking": mt_ranking_reward,
    "mt-da": mt_da_reward,
    "translation": translation_reward_da,
} # max reward: 1.0

UNIVERSAL_REWARDS = {
    "format": cumulative_format_reward, # 0.500
    "exact_format": exact_format_reward, # 0.250
    "unique_lines": unique_lines_reward, # 0.250
} # max reward: 1.0


def reward_router(data_source, solution_str, ground_truth, extra_info=None) -> float:
    reward = 0.0

    for reward_fn in UNIVERSAL_REWARDS.values():
        reward += reward_fn(solution_str.strip(), ground_truth, extra_info)

    if data_source in TASK_REWARDS:
        reward += TASK_REWARDS[data_source](solution_str.strip(), ground_truth, extra_info)

    #reward += reasoning_effort_reward(solution_str.strip(), ground_truth, extra_info)
        
    return reward
    
    