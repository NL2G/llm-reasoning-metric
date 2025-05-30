import pandas as pd
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_model_full_name(model_full_name: str) -> tuple[str, str]:
    """Parses 'model_base_name@[eval_kind]' into (model_base_name, eval_kind)."""
    if "@[" in model_full_name and model_full_name.endswith("]"):
        parts = model_full_name.rsplit("@[", 1)
        model_base_name = parts[0]
        eval_kind = parts[1][:-1]  # Remove trailing ']'
        return model_base_name, eval_kind
    logger.warning(f"Could not parse model_full_name according to 'name@[kind]' format: {model_full_name}")
    return model_full_name, "" # Return original name as base, and empty kind

def aggregate_results(output_dir_str: str = "llm-reasoning-metric/evals", output_csv_file: str = "aggregated_results_collection.csv"):
    root_path = Path(output_dir_str)
    all_metrics_data = []

    logger.info(f"Scanning for metrics.json files in {root_path}...")
    # Assumes structure: output_dir / experiment_folder / lang_pair_folder / metrics.json
    metric_files = list(root_path.glob('*/*/metrics.json'))
    logger.info(f"Found {len(metric_files)} metrics.json files.")

    if not metric_files:
        logger.info("No metrics.json files found. Exiting.")
        return

    for metric_file_path in metric_files:
        try:
            logger.debug(f"Processing {metric_file_path}...")
            with open(metric_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_full_name = data.pop("name", None)
            if not model_full_name:
                logger.warning(f"No 'name' field in {metric_file_path}. Skipping.")
                continue

            for metric_key, score in data.items():
                try:
                    key_parts = metric_key.split(' / ')
                    if len(key_parts) == 3:
                        lp_from_key = key_parts[0]
                        level = key_parts[1]
                        corr_fcn = key_parts[2]
                        
                        all_metrics_data.append({
                            "model_full_name": model_full_name,
                            "lp": lp_from_key,
                            "level": level,
                            "corr_fcn": corr_fcn,
                            "score": score
                        })
                    else:
                        logger.warning(f"Unexpected metric key format '{metric_key}' in {metric_file_path}. Skipping this key.")
                except Exception as e:
                    logger.error(f"Error processing metric key '{metric_key}' in {metric_file_path}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file {metric_file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing file {metric_file_path}: {e}")

    if not all_metrics_data:
        logger.info("No metric data extracted from files. Exiting.")
        return

    df_long = pd.DataFrame(all_metrics_data)
    
    if df_long.empty:
        logger.info("No data to process after loading. Exiting.")
        return

    # Add parsed model_name and kind
    parsed_name_kind = df_long["model_full_name"].apply(
        lambda x: pd.Series(parse_model_full_name(x), index=['model_base_name', 'eval_kind'])
    )
    df_long = pd.concat([df_long, parsed_name_kind], axis=1)

    final_results_list = []
    
    for model_full, group_df in df_long.groupby("model_full_name"):
        model_data_row = {"name": model_full}
        
        model_data_row["model_base_name"] = group_df["model_base_name"].iloc[0]
        model_data_row["eval_kind"] = group_df["eval_kind"].iloc[0]

        for _, row in group_df.iterrows():
            metric_col_name = f"{row['lp']}_{row['level']}_{row['corr_fcn']}"
            model_data_row[metric_col_name] = row['score']
        
        for (level, corr_fcn), type_specific_group_df in group_df.groupby(['level', 'corr_fcn']):
            avg_score = type_specific_group_df['score'].mean()
            avg_metric_col_name = f"avg_{level}_{corr_fcn}"
            model_data_row[avg_metric_col_name] = avg_score
            
        final_results_list.append(model_data_row)

    if not final_results_list:
        logger.info("No final results to save after processing. Exiting.")
        return

    df_final_wide = pd.DataFrame(final_results_list)
    
    core_cols = ['name', 'model_base_name', 'eval_kind']
    present_core_cols = [col for col in core_cols if col in df_final_wide.columns]
    
    other_cols = [col for col in df_final_wide.columns if col not in present_core_cols]
    lp_metric_cols = sorted([c for c in other_cols if not c.startswith('avg_')])
    avg_metric_cols = sorted([c for c in other_cols if c.startswith('avg_')])
    
    final_column_order = present_core_cols + lp_metric_cols + avg_metric_cols
    df_final_wide = df_final_wide.reindex(columns=final_column_order)

    logger.info(f"Saving aggregated results to {output_csv_file}")
    try:
        df_final_wide.to_csv(output_csv_file, index=False, float_format='%.4f')
        logger.info(f"Aggregation complete. Results saved to {Path(output_csv_file).resolve()}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")


if __name__ == "__main__":
    # This script should be run from the root of the llm-reasoning-metric project directory.
    # 'outputs' directory is expected to be at the root.
    # The resulting CSV will also be saved at the root.
    aggregate_results() 