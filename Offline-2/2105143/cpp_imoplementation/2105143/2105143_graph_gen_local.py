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
    0.75: "alpha=0.75/results.csv"  # Note: This file is named "results.csv" not "result.csv"
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
    if data_by_alpha[alpha].empty:
        print(f"Warning: Could not read data for alpha={alpha} from {file_paths[alpha]}")

# Ensure we have data for at least one alpha value
non_empty_alphas = [alpha for alpha, df in data_by_alpha.items() if not df.empty]

if non_empty_alphas:
    # Get common problems across available datasets
    common_problems = set(data_by_alpha[non_empty_alphas[0]]['Problem'])
    for alpha in non_empty_alphas[1:]:
        common_problems &= set(data_by_alpha[alpha]['Problem'])
    
    # Filter data to include only common problems and sort by problem number
    common_data = {}
    for alpha in non_empty_alphas:
        common_data[alpha] = data_by_alpha[alpha][data_by_alpha[alpha]['Problem'].isin(common_problems)].sort_values('Problem_Num')
    
    # Create comparison dataframe for Local Search (Average value)
    local_search_comparison = pd.DataFrame({'Problem': common_data[non_empty_alphas[0]]['Problem']})
    
    for alpha in non_empty_alphas:
        local_search_comparison[f'Alpha {alpha}'] = common_data[alpha]['Average value'].values
    
    # Create directory for output if it doesn't exist
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Plot for Local Search
    plt.figure(figsize=(16, 10))
    
    # Get problems as x-axis labels
    problems = local_search_comparison['Problem'].tolist()
    x = np.arange(len(problems))
    width = 0.25  # Width of each bar
    
    # Create bars for each alpha value
    colors = ['#3498db', '#e67e22', '#2ecc71']  # Blue, Orange, Green
    
    for i, alpha in enumerate(non_empty_alphas):
        position = x + (i - (len(non_empty_alphas) - 1) / 2) * width
        plt.bar(position, local_search_comparison[f'Alpha {alpha}'], width, 
                label=f'Alpha = {alpha}', color=colors[i % len(colors)])
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Local Search Value (Average value)', fontsize=14, fontweight='bold')
    plt.title('Local Search Performance with Different Alpha Values', fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    # Add grid lines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Calculate y-axis range with padding
    alpha_cols = [f'Alpha {alpha}' for alpha in non_empty_alphas]
    y_min = local_search_comparison[alpha_cols].min().min()
    y_max = local_search_comparison[alpha_cols].max().max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.1 * y_range)
    
    # Make a clean legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/local_search_alpha_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated Local Search comparison plot for different Alpha values")
    print("Plot saved in 'comparison_plots' directory")
    
    # Create a second plot with percentage differences
    plt.figure(figsize=(16, 10))
    
    # Calculate percentage differences using Alpha 0.25 as baseline (if available)
    baseline_alpha = non_empty_alphas[0]
    baseline_col = f'Alpha {baseline_alpha}'
    
    for alpha in non_empty_alphas[1:]:
        target_col = f'Alpha {alpha}'
        diff_col = f'Diff {alpha} vs {baseline_alpha}'
        local_search_comparison[diff_col] = ((local_search_comparison[target_col] - 
                                             local_search_comparison[baseline_col]) / 
                                             local_search_comparison[baseline_col] * 100)
    
    # Plot percentage differences
    for i, alpha in enumerate(non_empty_alphas[1:]):
        diff_col = f'Diff {alpha} vs {baseline_alpha}'
        plt.bar(x + (i - (len(non_empty_alphas) - 2) / 2) * width, 
                local_search_comparison[diff_col], width,
                label=f'Alpha {alpha} vs Alpha {baseline_alpha}', 
                color=colors[(i+1) % len(colors)])
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage Difference (%)', fontsize=14, fontweight='bold')
    plt.title(f'Local Search Performance Difference Compared to Alpha {baseline_alpha}', 
             fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    # Add grid lines and horizontal line at zero
    plt.grid(True, axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Make a clean legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/local_search_alpha_percentage_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated Local Search percentage difference plot")
    
    # Create a summary table
    summary_df = pd.DataFrame({
        'Problem': problems
    })
    
    # Add columns for each alpha value
    for alpha in non_empty_alphas:
        summary_df[f'Alpha {alpha}'] = local_search_comparison[f'Alpha {alpha}']
    
    # Add column for best alpha value for each problem
    best_alpha_cols = [f'Alpha {alpha}' for alpha in non_empty_alphas]
    summary_df['Best Alpha'] = summary_df[best_alpha_cols].idxmax(axis=1).apply(lambda x: x.split(' ')[1])
    
    # Calculate improvement of best over worst for each problem
    summary_df['Max Improvement (%)'] = (
        ((summary_df[best_alpha_cols].max(axis=1) - summary_df[best_alpha_cols].min(axis=1)) / 
         summary_df[best_alpha_cols].min(axis=1)) * 100
    )
    
    # Save the summary to CSV
    summary_df.to_csv('comparison_plots/local_search_alpha_summary.csv', index=False)
    print("Generated summary table for Local Search across alpha values")
    
else:
    print("Error: Could not read data from any of the specified files")