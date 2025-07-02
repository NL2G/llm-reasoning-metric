from rich import print
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
import hydra
from random import randint
import datasets as ds
from omegaconf import DictConfig
import logging
from rich.logging import RichHandler
from verl_reward import reward_router

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
    datefmt="[%X]",
)

logger = logging.getLogger(__name__)


def generate_answers(
    dataset: ds.Dataset,
    llm: LLM,
    sampling_params: SamplingParams,
) -> ds.Dataset:
    """
    Generate answers for a given dataset using a given LLM.
    """
    prompts = [item["prompt"] for item in dataset]
    outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
    example_idx: int = randint(0, len(prompts) - 1)
    logger.info("==="*20)
    logger.info(f"===> Example {example_idx} <===")
    logger.info("==="*20)
    logger.info(f"Prompt: {prompts[0]}")
    logger.info(f"Answer: {outputs[0].outputs[0].text}")
    logger.info(f"Ground Truth: {dataset[0]['reward_model']['ground_truth']}")
    reward = reward_router(
        data_source=dataset[0]['data_source'],
        solution_str=outputs[0].outputs[0].text,
        ground_truth=dataset[0]['reward_model']['ground_truth'],
        extra_info=dataset[0]['extra_info']
    )
    logger.info(f"Reward: {reward}")
    logger.info("==="*20)
    dataset = dataset.add_column("completion", [item.outputs[0].text for item in outputs])
    def get_reward(item: dict) -> dict[str, float]:
        return {
            "reward": reward_router(
                data_source=item['data_source'],
                solution_str=item['completion'],
                ground_truth=item['reward_model']['ground_truth'],
                extra_info=item['extra_info']
            )
        }
    
    if dataset[0]['data_source'] == "translation":
        logger.info("Evaluating translation data with 16 workers")
        dataset = dataset.map(get_reward, num_proc=16)
    else:
        logger.info("Evaluating completions with 1 worker")
        dataset = dataset.map(get_reward)
    
    logger.info(f"Rewards: {dataset.to_pandas()['reward'].describe()}")
    logger.info(f"Rewards: {dataset.to_pandas()['reward'].value_counts().reset_index().head(10)}")
    return dataset


@hydra.main(config_path="configs", config_name="sft-data")
def main(cfg: DictConfig):
    dataset = ds.Dataset.from_parquet(cfg.dataset.path)

    if cfg.dataset.n_shards > 0:
        logger.info(f"Sharding dataset into {cfg.dataset.n_shards} shards, shard {cfg.dataset.shard_id}")
        dataset = dataset.shard(num_shards=cfg.dataset.n_shards, index=cfg.dataset.shard_id, contiguous=True)

    llm = LLM(
        model=cfg.llm.model, 
        tokenizer_mode='mistral', 
        #config_format='mistral',
        max_model_len=cfg.llm.max_tokens + 1024,
        enable_prefix_caching=True,
        compilation_config=CompilationConfig(
            level=3
        )
    )
    sampling_params = SamplingParams(
        temperature=cfg.llm.temperature, 
        max_tokens=cfg.llm.max_tokens, 
        top_p=cfg.llm.top_p, 
        top_k=cfg.llm.top_k
    )
    if cfg.dataset.subsample > 0:
        dataset = dataset.select(range(cfg.dataset.subsample))

    dataset = generate_answers(dataset, llm, sampling_params)

    output_path = cfg.dataset.output_path
    
    dataset.to_parquet(output_path)

if __name__ == "__main__":
    main()