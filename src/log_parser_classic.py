"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script parses all log files to calculate final statistics for results table.
This only covers the classical models that dont need training (const_v, const_a, kinematics, xkalman)
Usage: python log_parser_classic.py 
Usage: python log_parser_classic.py [1]
    [1] - model_name ("const_v" or "const_a" or "kinematics" or "xkalman")
    
Example:
    python log_parser_classic.py const_v
"""


# #############################################################################
# ### IMPORTS
import os
import re
import sys




# #############################################################################
# ### MAIN
if __name__=="__main__":
    # Parse runargs
    args = sys.argv
    if len(args)!=2:
        print("Usage: python log_parser_classic.py model_name")
        sys.exit(0)
    model_name = args[1]
    
    # Path and models
    log_dir = "../data/3_logs/"
    model_prefix = model_name+"_"
    model_suffixes = ["25", "50", "75", "100"]
    file_template = model_prefix + "{}.txt"
    
    # Patterns for extracting ADE and FDE
    ade_pattern = re.compile(r">> ADE ([\d.]+) \[ ([\d.]+) \]")
    fde_pattern = re.compile(r">> FDE ([\d.]+) \[ ([\d.]+) \]")
    
    # Storage for results
    ade_means, ade_stds = [], []
    fde_means, fde_stds = [], []
    
    for suffix in model_suffixes:
        filepath = os.path.join(log_dir, file_template.format(suffix))
        with open(filepath, "r") as f:
            lines = f.readlines()
            # Find lines containing ADE and FDE
            ade_line = next(line for line in lines if line.startswith(">> ADE"))
            fde_line = next(line for line in lines if line.startswith(">> FDE"))
            # Extract values
            ade_match = ade_pattern.search(ade_line)
            fde_match = fde_pattern.search(fde_line)
            if ade_match and fde_match:
                ade_means.append(float(ade_match.group(1)))
                ade_stds.append(float(ade_match.group(2)))
                fde_means.append(float(fde_match.group(1)))
                fde_stds.append(float(fde_match.group(2)))
            else:
                raise ValueError(f"Could not parse ADE/FDE in {filepath}")
    
    # First row: means
    mean_row = f"| {model_name} "
    for val in ade_means:
        mean_row += f"| {val:.4f} "
    for val in fde_means:
        mean_row += f"| {val:.4f} "
    mean_row += "|"
    
    # Second row: stds in brackets
    std_row = "|       "
    for val in ade_stds:
        std_row += f"|[{val:.4f}]"
    for val in fde_stds:
        std_row += f"|[{val:.4f}]"
    std_row += "|"
    
    print("| Model  | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |")
    print("|------------|----|----|----|----|----|----|----|----|")
    print("| *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |")
    print(mean_row)
    print(std_row)