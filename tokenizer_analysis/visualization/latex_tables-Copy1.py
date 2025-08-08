"""
LaTeX table generation for tokenizer analysis results.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import os


class LaTeXTableGenerator:
    """Generate LaTeX tables from tokenizer analysis results."""
    
    def __init__(self, results: Dict[str, Any], tokenizer_names: List[str]):
        self.results = results
        self.tokenizer_names = tokenizer_names
        self.decimal_places = 3
        self.bold_best = True
    
    def set_formatting_options(self, decimal_places: int = 3, bold_best: bool = True):
        """Set formatting options."""
        self.decimal_places = decimal_places
        self.bold_best = bold_best
    
    def generate_basic_metrics_table(self, metrics: Optional[List[str]] = None) -> str:
        """Generate basic metrics comparison table."""
        if metrics is None:
            metrics = ['fertility', 'vocabulary_utilization', 'compression_ratio']
        
        # Table header
        header = "\\begin{tabular}{l" + "c" * len(self.tokenizer_names) + "}\n"
        header += "\\toprule\n"
        header += "Metric & " + " & ".join(self.tokenizer_names) + " \\\\\n"
        header += "\\midrule\n"
        
        rows = []
        
        for metric in metrics:
            if metric not in self.results:
                continue
                
            # Custom metric name mapping for display
            metric_names = {
                'compression_ratio': 'Compression Rate',
                'tokenizer_fairness_gini': 'Gini Coefficient',
                'vocabulary_utilization': 'Vocabulary Utilization'
            }
            row_name = metric_names.get(metric, metric.replace('_', ' ').title())
            row_values = []
            
            metric_data = self.results[metric].get('per_tokenizer', {})
            values_for_comparison = []
            
            for tok_name in self.tokenizer_names:
                if tok_name in metric_data:
                    if metric == 'fertility':
                        val = metric_data[tok_name]['global']['mean']
                    elif metric == 'vocabulary_utilization':
                        val = metric_data[tok_name]['global_utilization'] * 100
                    elif metric == 'compression_ratio':
                        val = metric_data[tok_name]['global']['mean']
                    else:
                        val = 0.0
                    
                    values_for_comparison.append(val)
                    formatted_val = f"{val:.{self.decimal_places}f}"
                    if metric == 'vocabulary_utilization':
                        formatted_val += "\\%"
                    row_values.append(formatted_val)
                else:
                    values_for_comparison.append(0.0)
                    row_values.append("--")
            
            # Bold the best value if requested
            if self.bold_best and values_for_comparison:
                if metric in ['fertility']:  # Lower is better
                    best_idx = np.argmin(values_for_comparison)
                else:  # Higher is better
                    best_idx = np.argmax(values_for_comparison)
                
                if best_idx < len(row_values):
                    row_values[best_idx] = f"\\textbf{{{row_values[best_idx]}}}"
            
            row = f"{row_name} & " + " & ".join(row_values) + " \\\\\n"
            rows.append(row)
        
        # Table footer
        footer = "\\bottomrule\n\\end{tabular}"
        
        return header + "".join(rows) + footer
    
    def generate_comprehensive_table(self, metrics: Optional[List[str]] = None) -> str:
        """Generate comprehensive table with multiple metrics."""
        if metrics is None:
            metrics = ['fertility', 'vocabulary_utilization', 'compression_ratio', 'morphscore']
        
        return self.generate_basic_metrics_table(metrics)
    
    def generate_information_theory_table(self, metrics: Optional[List[str]] = None) -> str:
        """Generate information theory metrics table."""
        if metrics is None:
            metrics = ['compression_ratio']
        
        return self.generate_basic_metrics_table(metrics)
    
    def generate_morphological_table(self, metrics: Optional[List[str]] = None) -> str:
        """Generate morphological metrics table."""
        if metrics is None:
            metrics = ['morphscore']
        
        return self.generate_basic_metrics_table(metrics)
    
    def save_table(self, table_content: str, output_path: str, caption: str = "", label: str = ""):
        """Save table to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            if caption:
                f.write(f"% Caption: {caption}\n")
            if label:
                f.write(f"% Label: {label}\n")
            f.write(table_content)