import pandas as pd
from pathlib import Path
import json
import logging
from rich.console import Console
from rich.table import Table

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup rich console
console = Console()

def parse_model_full_name(model_full_name: str) -> tuple[str, str]:
    """Parses 'model_base_name@[eval_kind]' into (model_base_name, eval_kind)."""
    if "@[" in model_full_name and model_full_name.endswith("]"):
        parts = model_full_name.rsplit("@[", 1)
        model_base_name = parts[0]
        eval_kind = parts[1][:-1]  # Remove trailing ']'
        return model_base_name, eval_kind
    logger.warning(f"Could not parse model_full_name according to 'name@[kind]' format: {model_full_name}")
    return model_full_name, "" # Return original name as base, and empty kind

def print_results_table(df: pd.DataFrame):
    """Print a rich-formatted table with the aggregated results."""
    if df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return
    
    # Create sorting keys for proper grouping
    def extract_model_info(row):
        model_base = row['model_base_name']
        eval_kind = row['eval_kind']
        
        # Extract reasoning effort tag if present (e.g., #none, #low, #medium, #high)
        reasoning_effort = ""
        if '#' in model_base:
            parts = model_base.split('#')
            if len(parts) == 2:
                reasoning_effort = parts[1]
                model_base = parts[0]  # Remove the reasoning effort tag from model_base
        
        # Extract base model name without MT-Eval prefix  
        if 'MT-Eval' in model_base:
            # Extract the base model (e.g., "Rexhaif/Qwen3-4B-MT-Eval" -> "Qwen3-4B")
            if 'Qwen3-14B' in model_base:
                base_model = 'Qwen3-14B'
            elif 'Qwen3-4B' in model_base:
                base_model = 'Qwen3-4B'
            else:
                base_model = model_base
            is_mt_eval = True
        else:
            # Extract base model for regular models (e.g., "Qwen/Qwen3-4B" -> "Qwen3-4B")
            if 'Qwen3-14B' in model_base:
                base_model = 'Qwen3-14B'
            elif 'Qwen3-4B' in model_base:
                base_model = 'Qwen3-4B'
            else:
                base_model = model_base
            is_mt_eval = False
            
        return base_model, eval_kind, is_mt_eval, reasoning_effort
    
    # Add sorting keys
    df_sorted = df.copy()
    model_info = df_sorted.apply(extract_model_info, axis=1, result_type='expand')
    df_sorted['_base_model'] = model_info[0]
    df_sorted['_eval_kind'] = model_info[1] 
    df_sorted['_is_mt_eval'] = model_info[2]
    df_sorted['_reasoning_effort'] = model_info[3]
    
    # Define eval_kind order
    eval_kind_order = {'gemba-da-like': 0, 'gemba-esa': 1, 'mt-ranking': 2}
    df_sorted['_eval_kind_order'] = df_sorted['_eval_kind'].map(eval_kind_order).fillna(999)
    
    # Define reasoning effort order: none, low, medium, high
    reasoning_effort_order = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, '': 4}  # Empty string for models without tags comes last
    df_sorted['_reasoning_effort_order'] = df_sorted['_reasoning_effort'].map(reasoning_effort_order).fillna(999)
    
    # Sort by: base_model, eval_kind_order, MT-Eval before regular (descending _is_mt_eval), then reasoning effort order
    df_sorted = df_sorted.sort_values(['_base_model', '_eval_kind_order', '_is_mt_eval', '_reasoning_effort_order'], 
                                      ascending=[True, True, False, True])
    
    # Create table
    table = Table(title="LLM Reasoning Metric Results (Grouped by Model Family)", show_header=True, header_style="bold magenta")
    
    # Add columns (excluding 'name')
    for col in df.columns:
        if col == 'name':
            continue  # Skip the name column
        elif col in ['model_base_name', 'eval_kind']:
            table.add_column(col, style="cyan", no_wrap=True)
        elif col.startswith('avg_'):
            table.add_column(col, style="bold green", justify="right")
        else:
            table.add_column(col, style="white", justify="right")
    
    # Add rows with group separation
    current_model = None
    current_eval_kind = None
    for _, row in df_sorted.iterrows():
        base_model = row['_base_model']
        eval_kind = row['_eval_kind']
        
        # Add separator line between different model sizes
        if current_model is not None and current_model != base_model:
            # Add empty row for visual separation between model sizes
            table.add_row(*["" for _ in df.columns if _ != 'name'], style="dim")
        # Add smaller separator between eval kinds within same model
        elif current_eval_kind is not None and current_eval_kind != eval_kind:
            # Add subtle separation between eval kinds
            table.add_row(*["" for _ in df.columns if _ != 'name'], style="dim blue")
        
        current_model = base_model
        current_eval_kind = eval_kind
        
        row_data = []
        for col in df.columns:
            if col == 'name':
                continue  # Skip the name column
            value = row[col]
            if pd.isna(value):
                row_data.append("—")
            elif isinstance(value, (int, float)):
                row_data.append(f"{value:.4f}")
            else:
                row_data.append(str(value))
        
        # Style MT-Eval rows slightly differently
        if row['_is_mt_eval']:
            table.add_row(*row_data, style="italic")
        else:
            table.add_row(*row_data)
    
    console.print("\n")
    console.print(table)
    console.print("\n")

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

    # Print rich-formatted table
    print_results_table(df_final_wide)
    
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