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
Usage: python log_parser_ml_multimodal_gmm.py [1]
    [1] - model_name ("social_lstm" or "social_bigat" or "physics_lstm" or "gatsbiv1" or "gatsbiv2")
    
Example:
    python log_parser_ml_multimodal_gmm.py social_lstm
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
            print(f"Error reading file {file_path}: {e}\n{content}")
            return None, None

# Function to process and compute average and STD for each experiment
def process_results(base_dir, model_name, prediction_lengths):      
    # Read results and process
    best_ade_means = {}
    best_experiments = {}
    
    # Iterate through each prediction length (25, 50, 75, 100)
    for prediction_length in prediction_lengths:
        best_ade_mean = float('inf')  # Start with a very large value
        best_experiment = None
    
        print(f"\nResults for prediction length: {prediction_length}")
        
        experiment_results = {
            i: {
                'ADE_a': [], 'ADE_b': [], 'ADE_c': [], 'ADE_d': [],
                'FDE_a': [], 'FDE_b': [], 'FDE_c': [], 'FDE_d': [],
            } for i in range(1, 50)
        }
        
        # Iterate through each split (1-5)
        for split in range(1, 6):
            for exp in range(1, 50):
                file_name = f"{model_name}_{prediction_length}_multimodal_gmm_split_{split}_{exp:02d}.model_perf.txt"
                file_path = os.path.join(base_dir, file_name)
                
                if os.path.exists(file_path):
                    ades, fdes = read_performance_file(file_path)
                    # if ades is None:
                    #     continue
                    # if len(ades)<4:
                    #     print("ERR", file_name)
                    #     continue
                    
                    if ades is not None and fdes is not None:
                        experiment_results[exp]['ADE_a'].append(ades[0])
                        experiment_results[exp]['ADE_b'].append(ades[1])
                        experiment_results[exp]['ADE_c'].append(ades[2])
                        experiment_results[exp]['ADE_d'].append(ades[3])
                        experiment_results[exp]['FDE_a'].append(fdes[0])
                        experiment_results[exp]['FDE_b'].append(fdes[1])
                        experiment_results[exp]['FDE_c'].append(fdes[2])
                        experiment_results[exp]['FDE_d'].append(fdes[3])
                else:
                    print(f"File not found: {file_path}")
        
        # Now calculate the average and STD for each experiment and track the best model
        for exp in range(1, 50):
            # Get all ADE and FDE lists for this experiment
            ade_a_values = experiment_results[exp]['ADE_a']
            ade_b_values = experiment_results[exp]['ADE_b']
            ade_c_values = experiment_results[exp]['ADE_c']
            ade_d_values = experiment_results[exp]['ADE_d']
            fde_a_values = experiment_results[exp]['FDE_a']
            fde_b_values = experiment_results[exp]['FDE_b']
            fde_c_values = experiment_results[exp]['FDE_c']
            fde_d_values = experiment_results[exp]['FDE_d']
            
            if ade_a_values and ade_b_values and ade_c_values and ade_d_values and fde_a_values and fde_b_values and fde_c_values and fde_d_values:
                ade_a_mean, ade_a_std = np.mean(ade_a_values), np.std(ade_a_values)
                ade_b_mean, ade_b_std = np.mean(ade_b_values), np.std(ade_b_values)
                ade_c_mean, ade_c_std = np.mean(ade_c_values), np.std(ade_c_values)
                ade_d_mean, ade_d_std = np.mean(ade_d_values), np.std(ade_d_values)
                fde_a_mean, fde_a_std = np.mean(fde_a_values), np.std(fde_a_values)
                fde_b_mean, fde_b_std = np.mean(fde_b_values), np.std(fde_b_values)
                fde_c_mean, fde_c_std = np.mean(fde_c_values), np.std(fde_c_values)
                fde_d_mean, fde_d_std = np.mean(fde_d_values), np.std(fde_d_values)
                
                print(
                    f"Experiment {exp:02d}: "
                    f"ADE_a: {ade_a_mean:.4f}±{ade_a_std:.4f}, "
                    f"ADE_b: {ade_b_mean:.4f}±{ade_b_std:.4f}, "
                    f"ADE_c: {ade_c_mean:.4f}±{ade_c_std:.4f}, "
                    f"ADE_c: {ade_d_mean:.4f}±{ade_d_std:.4f}, "
                    f"FDE_a: {fde_a_mean:.4f}±{fde_a_std:.4f}, "
                    f"FDE_b: {fde_b_mean:.4f}±{fde_b_std:.4f}, "
                    f"FDE_c: {fde_c_mean:.4f}±{fde_c_std:.4f}, "
                    f"FDE_d: {fde_d_mean:.4f}±{fde_d_std:.4f}"
                )
                
                # Track the best experiment with the lowest average ADE_b
                if ade_d_mean < best_ade_mean:
                    best_ade_mean = ade_d_mean
                    best_experiment = (
                        exp, prediction_length,
                        ade_a_mean, ade_a_std, ade_b_mean, ade_b_std, ade_c_mean, ade_c_std, ade_d_mean, ade_d_std,
                        fde_a_mean, fde_a_std, fde_b_mean, fde_b_std, fde_c_mean, fde_c_std, fde_d_mean, fde_d_std
                    )
        # Store the best model results for the markdown summary
        best_ade_means[prediction_length] = best_ade_mean
        best_experiments[prediction_length] = best_experiment
            
        
    # After processing all experiments, print the best experiment for each prediction length
    for key in best_ade_means:
        best_experiment = best_experiments[key]
        if best_experiment is not None:
            print("\n----------")
            print(f"Best Model for [{key}]: Experiment {best_experiment[0]:02d} with Prediction Length {best_experiment[1]} - Lowest Average ADE: {best_experiment[2]:.4f}")
            print(f"  ADE Mean: {best_experiment[4]:.4f}, ADE STD: {best_experiment[5]:.4f}")
            print(f"  FDE Mean: {best_experiment[12]:.4f}, FDE STD: {best_experiment[13]:.4f}")
        else:
            print("\nNo valid experiments found.")
    
    # Print the markdown summary of the best results based on the best models
    print("\n-----------")
    print("Markdown Summary of Best Results:")
    print("| Model  | Metric | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |")
    print("|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|")
    print("| | *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |")
    
    # The markdown summary now uses the best model results
        # ADE_a
    print(f"| {model_name} | ADE_a Mean | {best_experiments[25][2]:.4f} | {best_experiments[50][2]:.4f} | {best_experiments[75][2]:.4f} | {best_experiments[100][2]:.4f} | "
          f"{best_experiments[25][10]:.4f} | {best_experiments[50][10]:.4f} | {best_experiments[75][10]:.4f} | {best_experiments[100][10]:.4f} |")
    print(f"|        | ADE_a Std  | [{best_experiments[25][3]:.4f}] | [{best_experiments[50][3]:.4f}] | [{best_experiments[75][3]:.4f}] | [{best_experiments[100][3]:.4f}] | "
          f"[{best_experiments[25][11]:.4f}] | [{best_experiments[50][11]:.4f}] | [{best_experiments[75][11]:.4f}] | [{best_experiments[100][11]:.4f}] |")
        # ADE_b
    print(f"|        | ADE_b Mean | {best_experiments[25][4]:.4f} | {best_experiments[50][4]:.4f} | {best_experiments[75][4]:.4f} | {best_experiments[100][4]:.4f} | "
          f"{best_experiments[25][12]:.4f} | {best_experiments[50][12]:.4f} | {best_experiments[75][12]:.4f} | {best_experiments[100][12]:.4f} |")
    print(f"|        | ADE_b Std  | [{best_experiments[25][5]:.4f}] | [{best_experiments[50][5]:.4f}] | [{best_experiments[75][5]:.4f}] | [{best_experiments[100][5]:.4f}] | "
          f"[{best_experiments[25][13]:.4f}] | [{best_experiments[50][13]:.4f}] | [{best_experiments[75][13]:.4f}] | [{best_experiments[100][13]:.4f}] |")
        # ADE_c
    print(f"|        | ADE_c Mean | {best_experiments[25][6]:.4f} | {best_experiments[50][6]:.4f} | {best_experiments[75][6]:.4f} | {best_experiments[100][6]:.4f} | "
          f"{best_experiments[25][14]:.4f} | {best_experiments[50][14]:.4f} | {best_experiments[75][14]:.4f} | {best_experiments[100][14]:.4f} |")
    print(f"|        | ADE_c Std  | [{best_experiments[25][7]:.4f}] | [{best_experiments[50][7]:.4f}] | [{best_experiments[75][7]:.4f}] | [{best_experiments[100][7]:.4f}] | "
          f"[{best_experiments[25][15]:.4f}] | [{best_experiments[50][15]:.4f}] | [{best_experiments[75][15]:.4f}] | [{best_experiments[100][15]:.4f}] |")
        # ADE_d
    print(f"|        | ADE_d Mean | {best_experiments[25][8]:.4f} | {best_experiments[50][8]:.4f} | {best_experiments[75][8]:.4f} | {best_experiments[100][8]:.4f} | "
          f"{best_experiments[25][16]:.4f} | {best_experiments[50][16]:.4f} | {best_experiments[75][16]:.4f} | {best_experiments[100][16]:.4f} |")
    print(f"|        | ADE_d Std  | [{best_experiments[25][9]:.4f}] | [{best_experiments[50][9]:.4f}] | [{best_experiments[75][9]:.4f}] | [{best_experiments[100][9]:.4f}] | "
          f"[{best_experiments[25][17]:.4f}] | [{best_experiments[50][17]:.4f}] | [{best_experiments[75][17]:.4f}] | [{best_experiments[100][17]:.4f}] |")

# Main function
if __name__ == "__main__":
    # Parse runargs (no need for prediction_length as input anymore)
    args = sys.argv
    if len(args) != 2:
        print("Usage: python log_parser_ml_multimodal_gmm.py model_name")
        sys.exit(0)
    model_name = args[1]

    # Prediction lengths to process (25, 50, 75, 100)
    prediction_lengths = [25, 50, 75, 100]
    
    # Base directory where the model results are stored
    base_dir = "../data/4_models"
    
    # Read results and process
    process_results(base_dir, model_name, prediction_lengths)
