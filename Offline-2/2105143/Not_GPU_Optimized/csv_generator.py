import csv
import os
from collections import defaultdict
import numpy as np

from algorithms import (
    
    randomized_1,
    greedy_1,
    semi_greedy_1,
    local_search,
    grasp_max_cut
)

from file_input import(
    read_graphs
)

def generate_grasp_csv(input_dir="../../graph_GRASP/set2", output_file="results.csv", 
                     alpha=0.5, grasp_iterations=100, random_trials=100,local_search_runs=5):
    """
    Generates CSV results for all graph files in the input directory
    with the exact specified format.
    """
    # Prepare CSV header with merged cells structure
    header1 = [
        "Problem", "", "",
        "Constructive algorithm", "", "",
        "Local search", "",
        "GRASP", "",
        "Known best solution or upper bound"
    ]
    
    header2 = [
        "Name", "|V| or n", "|E| or m",
        "Simple Randomized or Randomized-1", "Simple Greedy or Greedy-1", "Semi-greedy-1",
        "Simple local or local-1", "",
        "GRASP-1", "",
        ""
    ]
    
    header3 = [
        "", "", "",
        "", "", "",
        "No. of iterations", "Average value",
        "No. of iterations", "Best value",
        ""
    ]

    # Prepare data rows
    data = []
    
    # Process all input files
    for filename in sorted(os.listdir(input_dir)):
        if filename.startswith("g") and filename.endswith(".rud"):
            file_path = os.path.join(input_dir, filename)
            problem_name = filename.split('.')[0]
            
            try:
                n, edges, graph = read_graphs(file_path)
                
                # Run all algorithms
                random_avg = randomized_1(random_trials, edges, graph)
                _, _, greedy_cut = greedy_1(edges, graph)
                _, _, semi_cut = semi_greedy_1(edges, graph, alpha)
                
                # Local search on semi-greedy solution
                local_cuts = []
                for _ in range(local_search_runs):
                    alpha = np.random.uniform(0, 1)  # Random alpha between 0 and 1
                    S,S_bar=set(),set()
                    S,S_bar,_=semi_greedy_1(edges, graph, alpha)
                    _, _, local_cut = local_search(S,S_bar, edges, graph)
                    local_cuts.append(local_cut)
                
                avg_local_cut = sum(local_cuts) / len(local_cuts)
                
                # GRASP with specified iterations
                _, _, grasp_cut = grasp_max_cut(edges, graph, alpha, grasp_iterations)
                
                # Known best
                all_values = [random_avg, greedy_cut, semi_cut, avg_local_cut, grasp_cut]
                max_value = max(all_values)
                
                known_best = max_value
                
                # Add row to data
                data.append([
                    problem_name, n, len(edges),
                    random_avg, greedy_cut, semi_cut,
                    local_search_runs, avg_local_cut,
                    grasp_iterations, grasp_cut,
                    known_best
                ])
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    # Write to CSV with multi-row header
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write merged headers
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerow(header3)
        
        # Write data rows
        writer.writerows(data)
    
    print(f"Results saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    generate_grasp_csv(
        input_dir="../graph_GRASP/set2",
        output_file="grasp_results.csv",
        alpha=0.5,
        grasp_iterations=100,
        random_trials=100
    )