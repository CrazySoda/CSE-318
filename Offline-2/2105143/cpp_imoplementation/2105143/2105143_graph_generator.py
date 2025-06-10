import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math

# Read the data from the file
with open('results.csv', 'r') as file:
    lines = file.readlines()

# Skip the header line and remove the last empty line
data_lines = [line for line in lines[1:-1] if line.strip()]

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

# Explicitly convert numerical columns to float
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

# Create a new dataframe for all problems (g1-g54)
plot_df = df[df['Problem'].str.startswith('g')].copy()

# Find max GRASP value for each problem
plot_df['GRASP_Max'] = plot_df[['GRASP-50 Val', 'GRASP-100 Val', 'GRASP-200 Val', 'GRASP-300 Val']].max(axis=1)

# Extract the problem number for sorting
plot_df['Problem_Num'] = plot_df['Problem'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
plot_df = plot_df.sort_values('Problem_Num')

# Function to create graphs with 10 problems each
def create_graph(data, start_idx, end_idx, filename):
    subset = data[(data['Problem_Num'] >= start_idx) & (data['Problem_Num'] <= end_idx)]
    
    # Set up the plot with white background
    plt.figure(figsize=(16, 10))
    plt.style.use('default')  # Use default style for white background
    
    # Set the x positions for the bars
    problems = subset['Problem'].tolist()
    x = np.arange(len(problems))
    width = 0.15  # Width of each bar
    
    # Create the bars
    plt.bar(x - width*2, subset['Randomized'], width, label='Randomized', color='#3498db')
    plt.bar(x - width, subset['Greedy'], width, label='Greedy', color='#e67e22')
    plt.bar(x, subset['Semi-greedy'], width, label='Semi-Greedy', color='#95a5a6')
    plt.bar(x + width, subset['GRASP_Max'], width, label='GRASP', color='#f1c40f')
    plt.bar(x + width*2, subset['Average value'], width, label='Local Search', color='#3498db', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=14, fontweight='bold')
    plt.title(f'Max Cut (Graph {start_idx}-{end_idx})', fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid lines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Calculate min and max values safely
    min_values = []
    max_values = []
    for col in ['Randomized', 'Greedy', 'Semi-greedy', 'GRASP_Max', 'Average value']:
        if not subset[col].empty and pd.api.types.is_numeric_dtype(subset[col]):
            min_values.append(subset[col].min())
            max_values.append(subset[col].max())
    
    # Set y-axis limits with padding
    min_val = min(min_values) if min_values else 0
    max_val = max(max_values) if max_values else 0
    
    # Adding 10% padding
    y_range = max_val - min_val
    plt.ylim(min_val - 0.1 * y_range, max_val + 0.1 * y_range)
    
    # Make a clean legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create multiple graphs, each with 10 problems
max_problem = plot_df['Problem_Num'].max()
num_graphs = math.ceil(max_problem / 10)

for i in range(num_graphs):
    start_idx = i * 10 + 1
    end_idx = min((i + 1) * 10, max_problem)
    create_graph(plot_df, start_idx, end_idx, f'max_cut_graph_{start_idx}-{end_idx}.png')

print(f"Generated {num_graphs} graphs covering all problems from g1 to g{max_problem}")
