import re
import math_verify as mv
import logging

logger = logging.getLogger(__name__)


def extract_answer(response) -> str | None:
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer:
        answer = answer.group(1).strip()
    else:
        answer = None
    return answer


def extract_thinking(response) -> str | None:
    thinking = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if thinking:
        thinking = thinking.group(1).strip()
    else:
        thinking = None
    return thinking


def count_xml(text) -> tuple[int, int]:
    count = 0
    extra_tokens = 0
    if text.count("<think>") == 1:
        count += 1
    if text.count("</think>") == 1:
        count += 1
    if text.count("<answer>") == 1:
        count += 1
    if text.count("</answer>") == 1:
        count += 1
        # penalize extra tokens after the answer tag
        extra_tokens = len(text.split("</answer>")[-1]) - 1
    return count, extra_tokens


def math_reward(solution_str, ground_truth, extra_info=None) -> float:
    answer = extract_answer(solution_str)
    if not answer:
        return 0.0
    
    format_regex = r"\\boxed{.*?}"
    match = re.search(format_regex, answer)
    if not match:
        return 0.0
    
    parsed_answer = mv.parse(answer, extraction_config=[mv.LatexExtractionConfig()])
    ground_truth = mv.parse(ground_truth, extraction_config=[mv.LatexExtractionConfig()])

    verification: bool = mv.verify(ground_truth, parsed_answer)
    
    if verification:
        return 1.0
    else:
        return 0.125 # format correctness reward


def extract_letter(text) -> tuple[bool | None, bool]:
    # Primary pattern: "Chosen: " followed by an uppercase letter
    pattern1 = r'Chosen: ([A-Z])'
    
    # Secondary pattern: Standalone uppercase letter (enclosed or not)
    # Matches: "A", <A>, [A], or just A
    # Must be surrounded by word boundaries or whitespace
    pattern2 = r'(?:^|(?<=\s))(?:"([A-Z])"|<([A-Z])>|\[([A-Z])\]|([A-Z]))(?:$|(?=\s))'
    
    # Try first pattern
    match = re.search(pattern1, text)
    if match:
        return match.group(1), True
    
    # Try second pattern
    match = re.search(pattern2, text, re.MULTILINE)
    if match:
        # Return whichever group matched (group 1, 2, 3, or 4)
        return (match.group(1) or match.group(2) or match.group(3) or match.group(4)), False
    
    return None, False
    

def mt_ranking_reward(solution_str, ground_truth, extra_info=None) -> float:
    parsed_answer = extract_answer(solution_str)
    if not parsed_answer:
        return 0.0
    
    answer, is_primary = extract_letter(parsed_answer)

    if not answer:
        return 0.0 
    
    if answer in {"A", "B"}:
        if answer == ground_truth:
            return 1.0 if is_primary else 0.8 # full correctness reward
        else:
            return 0.125 if is_primary else 0.075 # format correctness reward
    else:
        return 0.0

    

def cumulative_format_reward(solution_str, ground_truth, extra_info=None) -> float:
    reward = 0.0
    count, extra_tokens = count_xml(solution_str)
    if extra_tokens > 0:
        reward -= extra_tokens * 0.001
    
    reward += count * 0.125
    return reward

TASK_REWARDS = {
    "math": math_reward,
    "mt-ranking": mt_ranking_reward,
}

UNIVERSAL_REWARDS = {
    "format": cumulative_format_reward,
}


def reward_router(data_source, solution_str, ground_truth, extra_info=None) -> float:
    reward = 0.0

    for reward_fn in UNIVERSAL_REWARDS.values():
        reward += reward_fn(solution_str, ground_truth, extra_info)

    if data_source in TASK_REWARDS:
        reward += TASK_REWARDS[data_source](solution_str, ground_truth, extra_info)
        
    return reward
    
    