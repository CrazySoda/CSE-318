import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

# Define the alpha values and their file paths
alpha_values = [0.25, 0.5, 0.75]
file_paths = {
    0.25: "alpha=0.25/results.csv",
    0.5: "alpha=0.5/results.csv",
    0.75: "alpha=0.75/results.csv"
}

# Function to read and process the CSV files
def read_data(file_path):
    try:
        # Read the data from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Skip the header line and remove any empty lines
        data_lines = [line for line in lines[1:] if line.strip()]
        
        # Parse the CSV data
        data = []
        for line in data_lines:
            # Split by comma and clean up each field
            fields = [field.strip() for field in line.split(',')]
            data.append(fields)
        
        # Create DataFrame
        columns = ["Problem", "V", "E", "Randomized", "Greedy", "Semi-greedy", 
                   "Average value", "No. of iterations", 
                   "GRASP-50 Val", "Iterations-50", 
                   "GRASP-100 Val", "Iterations-100", 
                   "GRASP-200 Val", "Iterations-200", 
                   "GRASP-300 Val", "Iterations-300", 
                   "Best"]
        df = pd.DataFrame(data, columns=columns)
        
        # Convert numerical columns to float
        numeric_cols = ["Randomized", "Greedy", "Semi-greedy", "Average value", 
                        "GRASP-50 Val", "GRASP-100 Val", "GRASP-200 Val", "GRASP-300 Val"]
        
        for col in numeric_cols:
            # Convert to numeric and handle errors
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                # If there's an error, try to handle common issues
                df[col] = df[col].str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter for problems starting with 'g'
        plot_df = df[df['Problem'].str.startswith('g')].copy()
        
        # Find max GRASP value for each problem
        plot_df['GRASP_Max'] = plot_df[['GRASP-50 Val', 'GRASP-100 Val', 'GRASP-200 Val', 'GRASP-300 Val']].max(axis=1)
        
        # Extract the problem number for sorting
        plot_df['Problem_Num'] = plot_df['Problem'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        
        return plot_df.sort_values('Problem_Num')
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

# Read data for each alpha value
data_by_alpha = {}
for alpha in alpha_values:
    data_by_alpha[alpha] = read_data(file_paths[alpha])

# Ensure we have data for all alpha values
if all(not df.empty for df in data_by_alpha.values()):
    # Get common problems across all datasets
    common_problems = set(data_by_alpha[alpha_values[0]]['Problem'])
    for alpha in alpha_values[1:]:
        common_problems &= set(data_by_alpha[alpha]['Problem'])
    
    # Filter data to include only common problems and sort by problem number
    common_data = {}
    for alpha, df in data_by_alpha.items():
        common_data[alpha] = df[df['Problem'].isin(common_problems)].sort_values('Problem_Num')
    
    # Create comparison dataframes
    semi_greedy_comparison = pd.DataFrame({'Problem': common_data[alpha_values[0]]['Problem']})
    grasp_comparison = pd.DataFrame({'Problem': common_data[alpha_values[0]]['Problem']})
    
    for alpha in alpha_values:
        semi_greedy_comparison[f'Alpha {alpha}'] = common_data[alpha]['Semi-greedy'].values
        grasp_comparison[f'Alpha {alpha}'] = common_data[alpha]['GRASP_Max'].values
    
    # Create directory for output if it doesn't exist
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Plot for Semi-Greedy
    plt.figure(figsize=(16, 10))
    
    # Get problems as x-axis labels
    problems = semi_greedy_comparison['Problem'].tolist()
    x = np.arange(len(problems))
    width = 0.25  # Width of each bar
    
    # Create bars for each alpha value
    colors = ['#3498db', '#e67e22', '#2ecc71']  # Blue, Orange, Green
    
    for i, alpha in enumerate(alpha_values):
        plt.bar(x + (i-1)*width, semi_greedy_comparison[f'Alpha {alpha}'], width, 
                label=f'Alpha = {alpha}', color=colors[i])
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Cut Value', fontsize=14, fontweight='bold')
    plt.title('Semi-Greedy Algorithm Performance with Different Alpha Values', fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    # Add grid lines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Make a clean legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/semi_greedy_alpha_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot for GRASP
    plt.figure(figsize=(16, 10))
    
    # Create bars for each alpha value
    for i, alpha in enumerate(alpha_values):
        plt.bar(x + (i-1)*width, grasp_comparison[f'Alpha {alpha}'], width, 
                label=f'Alpha = {alpha}', color=colors[i])
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Cut Value', fontsize=14, fontweight='bold')
    plt.title('GRASP Algorithm Performance with Different Alpha Values', fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    # Add grid lines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Make a clean legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/grasp_alpha_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated comparison plots for Semi-Greedy and GRASP with different alpha values")
    print("Plots saved in 'comparison_plots' directory")
else:
    missing_files = [file_paths[alpha] for alpha in alpha_values if data_by_alpha[alpha].empty]
    print(f"Error: Missing or invalid data files: {missing_files}")