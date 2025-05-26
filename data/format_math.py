import datasets as ds
import argparse as ap
from rich import print
from rich_argparse import RichHelpFormatter


SYSTEM_PROMPT = """
You are a deep-thinking math expert.
You are given a math problem and you need to solve it through reasoning, step by step.
Use the following response format:
<think>
**your reasoning**
</think>
<answer>
\\boxed{**your answer**}
</answer>
""".strip()

def make_map_fn(problem_column: str, answer_column: str):
    
    def verl_format(item: dict, index: int):
        if "boxed" not in item[answer_column]:
            answer = "\\boxed{" + item[answer_column] + "}"
        else:
            answer = item[answer_column]

        return {
            "data_source": "math",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item[problem_column]}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "index": index
            }
        }
    
    return verl_format


def main():
    parser = ap.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--problem-column", type=str, default="problem")
    parser.add_argument("--answer-column", type=str, default="answer")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    dataset = ds.load_dataset(args.dataset_name)['train']
    columns = dataset.column_names
    map_fn = make_map_fn(args.problem_column, args.answer_column)
    
    verl_dataset = dataset.map(map_fn, num_proc=3, with_indices=True, remove_columns=columns)
    verl_dataset.to_parquet(args.output_path)

if __name__ == "__main__":
    main()