import re


SYSTEM_PROMPTS = {
    "math": """
You are a deep-thinking math expert.
You are given a math problem and you need to solve it through reasoning, step by step.
Use the following response format:
<think>
**your reasoning**
</think>
<answer>
\\boxed{**your answer**}
</answer>
""".strip(),
    "mt-ranking": """
You are a deep-thinking translation evaluator.
You are given a source text and a pair of translations, and you need to evaluate them and answer which one is better through reasoning, step by step.
Use the following response format:
<think>
**your reasoning**
</think>
<answer>
"Chosen: A" if Assistant A is better or "Chosen: B" if Assistant B is better
</answer>
""".strip(),
    "gemba-da-like": """
You are a deep-thinking translation evaluator.
You are given a source text and a translation, and you need to evaluate the quality of the translation through reasoning, step by step.
Use the following response format:
<think>
**your reasoning**
</think>
<answer>
**0 or 1 or 2 or ... or 100, depending on your judgement**
</answer>
""".strip()
}

USER_PROMPTS = {
    "math": """
{problem}
""".strip(),
    "mt-ranking": """
The user asked the two translation assistants, Assistant A and Assistant B, to translate the following source text from {source_language} to {target_language}, maintaining the tone, accuracy and ensuring fluency:
[Start of Source Text]
{source_text}
[End of Source Text]

[Start of Assistant A's Response]
{assistant_a_response}
[End of Assistant A's Response]

[Start of Assistant B's Response]
{assistant_b_response}
[End of Assistant B's Response]
""".strip(),
    "gemba-da-like": """
Score the following translation from {source_language} to {target_language} on a continuous scale from 0 to 100,
where a score of zero (0) means "no meaning preserved" and score of one hundred (100) means "perfect meaning and grammar".

[Start of Source Text]
{source_text}
[End of Source Text]

[Start of Translation]
{translation}
[End of Translation]
""".strip()
}

LANG_CODES = {
    'en': 'English',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'cs': 'Czech',
    'uk': 'Ukrainian',
    'de': 'German',
    'es': 'Spanish',
    'he': 'Hebrew',
    'ar': 'Arabic',
    'ru': 'Russian',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
}

def parse_numerical_answer(answer, min=None, max=None):
    # get all numbers in a string
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 1:
        return int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    return None


def parse_and_check_numerical_answer(answer, min=None, max=None):
    attempt = parse_numerical_answer(answer, min, max)
    if attempt is not None:
        if attempt < min or attempt > max:
            return None
        return attempt

    return None
