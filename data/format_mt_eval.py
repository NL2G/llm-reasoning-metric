import datasets as ds
import argparse as ap
from rich import print
from rich_argparse import RichHelpFormatter


SYSTEM_PROMPT = """
You are a deep-thinking translation evaluator.
You are given a source text and a pair of translations, and you need to evaluate them and answer which one is better through reasoning.
Use the following response format:
<think>
**your reasoning**
</think>
<answer>
[A] if Assistant A is better, [B] if Assistant B is better
</answer>
""".strip()

USER_PROMPT_TEMPLATE = """
The user asked the two translation assistants, Assistant A and Assistant B, to translate the following source text from {source_language} to {target_language}:
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

LANG_CODES = {
    'en': 'English',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'cs': 'Czech',
    'uk': 'Ukrainian',
    'de': 'German',
    'es': 'Spanish'
}


def main():
    parser = ap.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True, help="subset of the dataset to use, ALL for all subsets")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    dataset = ds.load_dataset(args.dataset_name)['train']
    if args.subset != 'ALL':
        dataset = dataset.filter(lambda x: x['source'] == args.subset)

    columns = dataset.column_names

    def mt_ranking_to_verl_format(item: dict, index: int):
        lang1, lang2 = item['lp'].split('-')
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
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
                "index": index
            }
        }
    
    verl_dataset = dataset.map(mt_ranking_to_verl_format, num_proc=1, with_indices=True, remove_columns=columns)
    verl_dataset.to_parquet(args.output_path)

if __name__ == "__main__":
    main()