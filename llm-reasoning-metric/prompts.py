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