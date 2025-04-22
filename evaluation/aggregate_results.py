#!/usr/bin/env python
import os
import json
import glob
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate evaluation results into a single table')
    parser.add_argument('--input-dir', type=str, default='.',
                        help='Directory containing evaluation JSON files')
    parser.add_argument('--output-csv', type=str, default='aggregated_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['sys / pce', 'seg / KendallWithTiesOpt'],
                        help='Metrics to include in the aggregation')
    parser.add_argument('--lang-pairs', type=str, nargs='+',
                        default=['en-de', 'en-es', 'ja-zh'],
                        help='Language pairs to include in the aggregation')
    return parser.parse_args()

def extract_model_info(filename):
    """Extract model type, size, and other info from filename."""
    basename = os.path.basename(filename)
    basename = basename.replace('.json', '')
    parts = basename.split('-')
    
    # Handle different naming patterns
    if len(parts) >= 3:
        if 'simple' in basename:
            # Handle files like qwen25-1-5b-simple.json
            return basename, '-'.join(parts[:-1]) if 'simple' in parts[-1] else '-'.join(parts)
        else:
            # Handle files like mt-ranker-1.5b-en-de.json
            model_type = parts[0]
            if len(parts) > 3 and parts[1] == 'ranker':
                model_type = f"{parts[0]}-{parts[1]}"
                model_size = parts[2]
                return f"{model_type}-{model_size}", model_type
            else:
                model_type = parts[0]
                model_size = parts[1]
                return f"{model_type}-{model_size}", model_type
    return basename, basename  # Fallback

def aggregate_results(input_dir, metrics, lang_pairs):
    """Aggregate results from all JSON files."""
    result_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    # Filter out trace files
    result_files = [f for f in result_files if 'traces' not in f]
    
    results = {}
    model_types = set()
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        model_id, model_type = extract_model_info(file_path)
        model_name = data.get('name', '') or model_id
        model_types.add(model_type)
        
        if model_name not in results:
            results[model_name] = {}
        
        # Extract metrics for each language pair
        for lang_pair in lang_pairs:
            for metric in metrics:
                key = f"{lang_pair} / {metric}"
                if key in data:
                    results[model_name][key] = data[key]

    return results, model_types

def create_table(results, metrics, lang_pairs):
    """Create a rich table from the aggregated results."""
    console = Console()
    table = Table(title="Aggregated Evaluation Results")
    
    # Add columns
    table.add_column("Model", style="cyan")
    
    # Add columns for each metric and language pair
    for lang_pair in lang_pairs:
        for metric in metrics:
            table.add_column(f"{lang_pair} / {metric}", style="green")
    
    # Add rows for each model
    for model_name, model_data in results.items():
        row = [model_name]
        for lang_pair in lang_pairs:
            for metric in metrics:
                key = f"{lang_pair} / {metric}"
                value = model_data.get(key, "--")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                row.append(str(value))
        table.add_row(*row)
    
    return table

def create_dataframe(results, metrics, lang_pairs):
    """Create a pandas DataFrame from the aggregated results."""
    data = []
    
    for model_name, model_data in results.items():
        row = {'Model': model_name}
        for lang_pair in lang_pairs:
            for metric in metrics:
                key = f"{lang_pair} / {metric}"
                value = model_data.get(key, "--")
                # Format float values to have 4 decimal places
                if isinstance(value, float):
                    value = round(value, 4)
                row[key] = value
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    args = parse_args()
    
    results, model_types = aggregate_results(args.input_dir, args.metrics, args.lang_pairs)
    
    # Create and display the table
    table = create_table(results, args.metrics, args.lang_pairs)
    console = Console()
    console.print(table)
    
    # Create and save the DataFrame
    df = create_dataframe(results, args.metrics, args.lang_pairs)
    df.to_csv(args.output_csv, index=False)
    console.print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main() 