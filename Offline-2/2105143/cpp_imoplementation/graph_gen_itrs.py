import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

# Define the file path
file_path = "alpha=0.5/results.csv"

# Function to read and process the CSV file
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

# Read data from the file
data = read_data(file_path)

if not data.empty:
    # Create directory for output if it doesn't exist
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Get problems as x-axis labels
    problems = data['Problem'].tolist()
    x = np.arange(len(problems))
    width = 0.2  # Width of each bar
    
    # Create figure with larger size
    plt.figure(figsize=(18, 10))
    
    # Define colors and iteration values
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']  # Blue, Red, Green, Purple
    iterations = [50, 100, 200, 300]
    
    # Create bars for each iteration
    for i, iter_val in enumerate(iterations):
        plt.bar(x + (i-1.5)*width, data[f'GRASP-{iter_val} Val'], width, 
                label=f'GRASP-{iter_val}', color=colors[i])
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Cut Value', fontsize=14, fontweight='bold')
    plt.title('GRASP Performance with Different Iteration Values (Alpha=0.75)', fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    # Add grid lines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Calculate y-axis range with padding
    y_min = data[['GRASP-50 Val', 'GRASP-100 Val', 'GRASP-200 Val', 'GRASP-300 Val']].min().min()
    y_max = data[['GRASP-50 Val', 'GRASP-100 Val', 'GRASP-200 Val', 'GRASP-300 Val']].max().max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.1 * y_range)
    
    # Add legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Add a text box with information about alpha value
    plt.text(0.02, 0.98, "Alpha = 0.75", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/grasp_iterations_comparison_alpha_0.75.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated GRASP iterations comparison plot for Alpha=0.75")
    print("Plot saved in 'comparison_plots' directory")
    
    # Create a second graph showing percentage improvement over iterations
    plt.figure(figsize=(18, 10))
    
    # Calculate percentage improvement using GRASP-50 as baseline
    for iter_val in [100, 200, 300]:
        data[f'Improvement-{iter_val}'] = ((data[f'GRASP-{iter_val} Val'] - data['GRASP-50 Val']) / 
                                        data['GRASP-50 Val'] * 100)
    
    # Plot improvement percentages
    plt.bar(x - width, data['Improvement-100'], width, label='GRASP-100 vs GRASP-50', color='#e74c3c')
    plt.bar(x, data['Improvement-200'], width, label='GRASP-200 vs GRASP-50', color='#2ecc71')
    plt.bar(x + width, data['Improvement-300'], width, label='GRASP-300 vs GRASP-50', color='#9b59b6')
    
    # Add labels and title
    plt.xlabel('Problem', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage Improvement (%)', fontsize=14, fontweight='bold')
    plt.title('Percentage Improvement in GRASP Performance with More Iterations (Alpha=0.75)', 
             fontsize=18, fontweight='bold')
    plt.xticks(x, problems, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    # Add grid lines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Add a text box with information about alpha value
    plt.text(0.02, 0.98, "Alpha = 0.75", transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/grasp_iterations_improvement_alpha_0.75.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated GRASP iterations improvement comparison plot for Alpha=0.75")
    
else:
    print(f"Error: Could not read data from {file_path}")