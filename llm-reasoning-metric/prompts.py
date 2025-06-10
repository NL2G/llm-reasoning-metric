import re


SYSTEM_PROMPTS = {
    "math": """
You are a deep-thinking math expert.
You are given a math problem and you need to solve it through reasoning, step by step.
Use the following response format:
<think>
'your reasoning'
</think>
<answer>
\\boxed{your answer}
</answer>
""".strip(),
    "mt-ranking": """
You are a deep-thinking translation evaluator.
You are given a source text and a pair of translations, and you need to evaluate them and answer which one is better through reasoning, step by step.
Use the following response format:
<think>
'your reasoning'
</think>
<answer>
'your answer'
</answer>
""".strip(),
    "gemba-da-like": """
You are a deep-thinking translation evaluator.
You are given a source text and a translation, and you need to evaluate the quality of the translation through reasoning, step by step.
Use the following response format:
<think>
'your reasoning'
</think>
<answer>
'your answer'
</answer>
""".strip(),
    "gemba-esa": """
Your task is to identify machine translation errors and assess the quality of the translation.
""".strip(),
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

If Assistant A's response is better than Assistant B's response, answer 'Chosen: A', otherwise answer 'Chosen: B'.
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

Your answer should be a number like 0 or 1 or 2 or ... or 100, depending on your judgement.
""".strip(),
    "gemba-esa-error-spans": """
{source_language} source:
```{source_text}```
{target_language} translation:
```{translation}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
Each error is classified as one of two categories: major or minor. Major errors disrupt the flow and make the understandability of text difficult or impossible. Minor errors are errors that do not disrupt the flow significantly and what the text is trying to say is still understandable.
""".strip(),
    "gemba-esa-ranking": """
Given the translation from {source_language} to {target_language} and the annotated error spans, assign a score on a continuous scale from 0 to 100.
The scale has following reference points: 0="No meaning preserved", 33="Some meaning preserved", 66="Most meaning preserved and few grammar mistakes", up to 100="Perfect meaning and grammar".

Score the following translation from {source_language} source:
```{source_text}```
{target_language} translation:
```{translation}```
Annotated error spans:
```{error_spans}```
""".strip(),
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
    'hr': 'Croatian',
    'liv': 'Livonian',
    'hi': 'Hindi',
    'is': 'Icelandic',
}

GEMBA_FEW_SHOTS = [
    {
        "source_language": "English",
        "target_language": "German",
        "source_text": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
        "translation": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
        "response": """
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""".strip()
    },
    {
        "source_language": "English",
        "target_language": "Czech",
        "source_text": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
        "translation": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
        "response": """
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""".strip()
    },
    {
        "source_language": "Chinese",
        "target_language": "English",
        "source_text": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
        "translation": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
        "response": """
Major:
accuracy/addition - "of high-speed rail"
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""".strip()
    }
]

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


