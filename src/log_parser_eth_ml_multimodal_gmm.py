"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script parses all log files to calculate final statistics for results table.
This only covers the ML models that have been trained for 50 epochs on 5 different test-training splits.
Usage: python log_parser_ml_unimodal.py [1]
    [1] - model_name ("social_lstm" or "social_bigat" or "gatsbi")
    [2] - source ("ETH" or "HOTEL")
Example:
    python log_parser_eth_ml_unimodal.py social_lstm ETH
"""



# #############################################################################
# ### IMPORTS
import os
import json
import numpy as np
import sys




# #############################################################################
# ### METHODS

# Function to read the content of a .txt file and extract ADE, FDE
def read_performance_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        content = content.replace("'", '"')
        try:
            performance = json.loads(content)
            return performance['ADE'], performance['FDE']
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None, None

# Function to process and compute average and STD for each experiment
def process_results(base_dir, model_name, source, prediction_lengths):          
    # Read results and process
    best_ade_means = {}
    best_experiments = {}
    
    # Iterate through each prediction length (25, 50, 75, 100)
    for prediction_length in prediction_lengths:
        best_ade_mean = float('inf')  # Start with a very large value
        best_experiment = None
    
        print(f"\nResults for prediction length: {prediction_length}")
        
        experiment_results = {i: {'ADE': [], 'FDE': []} for i in range(1, 50)}
        
        # Iterate through each split (1-5)
        for exp in range(1, 50):
            file_name = f"{model_name}_{prediction_length}_multimodal_gmm_{source}_{exp:02d}.model_perf.txt"
            file_path = os.path.join(base_dir, file_name)
            if os.path.exists(file_path):
                ade, fde = read_performance_file(file_path)
                ade = ade[-1]
                fde = fde[-1]
                if ade is not None and fde is not None:
                    experiment_results[exp]['ADE'].append(ade)
                    experiment_results[exp]['FDE'].append(fde)
            else:
                print(f"File not found: {file_path}")
        
        # Now calculate the average and STD for each experiment and track the best model
        for exp in range(1, 50):
            ade_values = experiment_results[exp]['ADE']
            fde_values = experiment_results[exp]['FDE']
                        
            if ade_values and fde_values:
                ade_mean = np.mean(ade_values)
                ade_std = np.std(ade_values)
                fde_mean = np.mean(fde_values)
                fde_std = np.std(fde_values)
                print(f"Experiment {exp:02d}:  ADE - Mean: {ade_mean:.4f}, STD: {ade_std:.4f}  FDE - Mean: {fde_mean:.4f}, STD: {fde_std:.4f}")
                # Track the best experiment with the lowest average ADE
                if ade_mean < best_ade_mean:
                    best_ade_mean = ade_mean
                    best_experiment = (exp, prediction_length, ade_mean, ade_std, fde_mean, fde_std)
        
        # Store the best model results for the markdown summary
        best_ade_means[prediction_length] = best_ade_mean
        best_experiments[prediction_length] = best_experiment
        
    # After processing all experiments, print the best experiment for each prediction length
    for key in best_ade_means:
        best_experiment = best_experiments[key]
        if best_experiment is not None:
            print("\n----------")
            print(f"Best Model for [{key}]: Experiment {best_experiment[0]:02d} with Prediction Length {best_experiment[1]} - Lowest Average ADE: {best_experiment[2]:.4f}")
            print(f"  ADE Mean: {best_experiment[2]:.4f}, ADE STD: {best_experiment[3]:.4f}")
            print(f"  FDE Mean: {best_experiment[4]:.4f}, FDE STD: {best_experiment[5]:.4f}")
        else:
            print("\nNo valid experiments found.")
    
    # Print the markdown summary of the best results based on the best models
    print("\n-----------")
    print("Markdown Summary of Best Results:")
    print("| Model  | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |")
    print("|--------|-----|-----|-----|-----|-----|-----|-----|-----|")
    print("| *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |")
    
    # The markdown summary now uses the best model results
    print(f"| {model_name} | {best_experiments[2][2]:.4f} | {best_experiments[4][2]:.4f} | {best_experiments[6][2]:.4f} | {best_experiments[10][2]:.4f} | {best_experiments[2][4]:.4f} | {best_experiments[4][4]:.4f} | {best_experiments[6][4]:.4f} | {best_experiments[10][4]:.4f} |")
    print(f"|       | [{best_experiments[2][3]:.4f}] | [{best_experiments[4][3]:.4f}] | [{best_experiments[6][3]:.4f}] | [{best_experiments[10][3]:.4f}] | [{best_experiments[2][5]:.4f}] | [{best_experiments[4][5]:.4f}] | [{best_experiments[6][5]:.4f}] | [{best_experiments[10][5]:.4f}] |")

# Main function
if __name__ == "__main__":
    # Parse runargs (no need for prediction_length as input anymore)
    args = sys.argv
    if len(args) != 2:
        print("Usage: python log_parser_eth_ml_unimodal.py model_name source")
        sys.exit(0)
    model_name = args[1]
    source = args[2]

    # Prediction lengths to process (2, 4, 6, 10)
    prediction_lengths = [2, 4, 6, 10]
    
    # Base directory where the model results are stored
    base_dir = "../data/4_models"
    
    # Read results and process
    process_results(base_dir, model_name, source, prediction_lengths)
