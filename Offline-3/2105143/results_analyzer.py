#!/usr/bin/env python3
"""
Results Analyzer and Report Generator
Analyzes experiment results and generates tables/charts for assignment report
"""

import json
import csv
import os
import warnings
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

# Handle optional dependencies gracefully
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Some features will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    HAS_MATPLOTLIB = True
    
    # Set backend before importing seaborn
    matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
        print("Warning: seaborn not installed. Using basic matplotlib styling.")
        
except ImportError:
    HAS_MATPLOTLIB = False
    HAS_SEABORN = False
    print("Warning: matplotlib not installed. No visualizations will be generated.")

class ResultsAnalyzer:
    """Analyzes experiment results and generates visualizations."""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = results_dir
        self.task4_data = None
        self.task5_data = None
        
        # Set up plotting style if available
        if HAS_MATPLOTLIB:
            try:
                # Try different style options
                available_styles = plt.style.available
                if 'seaborn' in available_styles:
                    plt.style.use('seaborn')
                elif 'seaborn-v0_8' in available_styles:
                    plt.style.use('seaborn-v0_8')
                elif 'default' in available_styles:
                    plt.style.use('default')
                
                if HAS_SEABORN:
                    sns.set_palette("husl")
                    
            except Exception as e:
                print(f"Warning: Could not set plotting style: {e}")
                # Continue with default style
    
    def load_latest_results(self):
        """Load the most recent experiment results."""
        if not HAS_PANDAS:
            print("❌ pandas is required for result analysis. Please install with: pip install pandas")
            return False
            
        try:
            if not os.path.exists(self.results_dir):
                print(f"❌ Results directory '{self.results_dir}' not found!")
                return False
                
            # Find latest files
            all_files = os.listdir(self.results_dir)
            task4_files = [f for f in all_files if f.startswith('task4_summary_') and f.endswith('.csv')]
            task5_files = [f for f in all_files if f.startswith('task5_summary_') and f.endswith('.csv')]
            
            if task4_files:
                latest_task4 = max(task4_files)
                self.task4_data = pd.read_csv(os.path.join(self.results_dir, latest_task4))
                print(f"✅ Loaded Task 4 data: {latest_task4} ({len(self.task4_data)} records)")
            else:
                print("ℹ️  No Task 4 results found")
            
            if task5_files:
                latest_task5 = max(task5_files)
                self.task5_data = pd.read_csv(os.path.join(self.results_dir, latest_task5))
                print(f"✅ Loaded Task 5 data: {latest_task5} ({len(self.task5_data)} records)")
            else:
                print("ℹ️  No Task 5 results found")
                
            return True
                
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def generate_task4_analysis(self):
        """Generate Task 4 analysis and visualizations."""
        if not HAS_PANDAS or self.task4_data is None:
            print("❌ No Task 4 data available or pandas not installed!")
            return
        
        print("\n📊 GENERATING TASK 4 ANALYSIS...")
        
        try:
            # 1. Depth Analysis Chart
            depth_data = self.task4_data[self.task4_data['Experiment_Type'] == 'depth_analysis']
            if not depth_data.empty:
                if HAS_MATPLOTLIB:
                    self._plot_depth_analysis(depth_data)
                self._create_depth_table(depth_data)
            
            # 2. Timeout Analysis Chart
            timeout_data = self.task4_data[self.task4_data['Experiment_Type'] == 'timeout_analysis']
            if not timeout_data.empty:
                if HAS_MATPLOTLIB:
                    self._plot_timeout_analysis(timeout_data)
                self._create_timeout_table(timeout_data)
            
            # 3. Heuristic Comparison
            heuristic_data = self.task4_data[self.task4_data['Experiment_Type'] == 'heuristic_analysis']
            if not heuristic_data.empty:
                if HAS_MATPLOTLIB:
                    self._plot_heuristic_comparison(heuristic_data)
                self._create_heuristic_table(heuristic_data)
                
        except Exception as e:
            print(f"❌ Error in Task 4 analysis: {e}")
    
    def generate_task5_analysis(self):
        """Generate Task 5 analysis and visualizations."""
        if not HAS_PANDAS or self.task5_data is None:
            print("❌ No Task 5 data available or pandas not installed!")
            return
        
        print("\n📊 GENERATING TASK 5 ANALYSIS...")
        
        try:
            # 1. Tournament Matrix
            tournament_data = self.task5_data[self.task5_data['Experiment_Type'] == 'tournament']
            if not tournament_data.empty:
                if HAS_MATPLOTLIB:
                    self._plot_tournament_matrix(tournament_data)
                self._create_tournament_table(tournament_data)
            
            # 2. Performance vs Random
            vs_random_data = self.task5_data[self.task5_data['Experiment_Type'] == 'vs_random']
            if not vs_random_data.empty:
                if HAS_MATPLOTLIB:
                    self._plot_vs_random_performance(vs_random_data)
                self._create_performance_table(vs_random_data)
            
            # 3. Overall Rankings
            self._create_overall_rankings()
            
        except Exception as e:
            print(f"❌ Error in Task 5 analysis: {e}")
    
    def _plot_depth_analysis(self, data):
        """Plot depth vs performance analysis."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Task 4: Search Depth Analysis (vs Random Agent)', fontsize=16, fontweight='bold')
            
            # Extract depth from parameter names and sort
            data = data.copy()
            data['Depth'] = data['Parameter'].str.extract(r'depth_(\d+)')[0].astype(int)
            data = data.sort_values('Depth')
            
            # Win Rate vs Depth
            ax1.plot(data['Depth'], data['P2_Win_Rate'], 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Search Depth')
            ax1.set_ylabel('Win Rate vs Random')
            ax1.set_title('A) Win Rate vs Search Depth')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Average Moves vs Depth
            ax2.plot(data['Depth'], data['Avg_Moves'], 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Search Depth')
            ax2.set_ylabel('Average Moves per Game')
            ax2.set_title('B) Game Length vs Search Depth')
            ax2.grid(True, alpha=0.3)
            
            # Nodes Explored vs Depth (with error handling for missing data)
            if 'Avg_Nodes_P2' in data.columns and not data['Avg_Nodes_P2'].isna().all():
                ax3.semilogy(data['Depth'], data['Avg_Nodes_P2'], 'go-', linewidth=2, markersize=8)
                ax3.set_xlabel('Search Depth')
                ax3.set_ylabel('Average Nodes Explored (log scale)')
                ax3.set_title('C) Computational Cost vs Search Depth')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Node data not available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('C) Computational Cost (Data Missing)')
            
            # Performance Efficiency
            if 'Avg_Nodes_P2' in data.columns and not data['Avg_Nodes_P2'].isna().all():
                efficiency = data['P2_Win_Rate'] / (data['Avg_Nodes_P2'] / 1000 + 1e-6)  # Avoid division by zero
                ax4.plot(data['Depth'], efficiency, 'mo-', linewidth=2, markersize=8)
                ax4.set_ylabel('Efficiency (Win Rate / Computation)')
            else:
                ax4.plot(data['Depth'], data['P2_Win_Rate'], 'mo-', linewidth=2, markersize=8)
                ax4.set_ylabel('Win Rate')
            
            ax4.set_xlabel('Search Depth')
            ax4.set_title('D) Performance Efficiency vs Depth')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'task4_depth_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Depth analysis chart saved: {save_path}")
            plt.close()  # Close to free memory
            
        except Exception as e:
            print(f"❌ Error creating depth analysis plot: {e}")
    
    def _plot_timeout_analysis(self, data):
        """Plot timeout analysis."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Task 4: Timeout Analysis (Balanced vs Aggressive)', fontsize=16, fontweight='bold')
            
            # Extract timeout values and sort
            data = data.copy()
            data['Timeout'] = data['Parameter'].str.extract(r'timeout_(\d+\.?\d*)')[0].astype(float)
            data = data.sort_values('Timeout')
            
            # Win rates over time
            ax1.plot(data['Timeout'], data['P1_Win_Rate'], 'b-o', label='Balanced', linewidth=2)
            ax1.plot(data['Timeout'], data['P2_Win_Rate'], 'r-o', label='Aggressive', linewidth=2)
            ax1.set_xlabel('Timeout (seconds)')
            ax1.set_ylabel('Win Rate')
            ax1.set_title('Win Rates vs Timeout')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Game duration vs timeout
            ax2.plot(data['Timeout'], data['Avg_Duration'], 'g-o', linewidth=2)
            ax2.set_xlabel('Timeout (seconds)')
            ax2.set_ylabel('Average Game Duration (seconds)')
            ax2.set_title('Game Duration vs Timeout')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'task4_timeout_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Timeout analysis chart saved: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating timeout analysis plot: {e}")
    
    def _plot_heuristic_comparison(self, data):
        """Plot heuristic comparison."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Task 4: Heuristic Effectiveness Analysis', fontsize=16, fontweight='bold')
            
            # Sort by win rate
            data_sorted = data.sort_values('P2_Win_Rate', ascending=True)
            
            # Create color palette
            n_colors = len(data_sorted)
            if HAS_MATPLOTLIB:
                colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
            else:
                colors = ['lightcoral', 'gold', 'lightgreen', 'skyblue'][:n_colors]
            
            # Horizontal bar chart of win rates
            bars = ax1.barh(data_sorted['Parameter'], data_sorted['P2_Win_Rate'], color=colors)
            ax1.set_xlabel('Win Rate vs Random')
            ax1.set_title('A) Heuristic Win Rates')
            ax1.set_xlim(0, 1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            # Average moves comparison
            bars2 = ax2.barh(data_sorted['Parameter'], data_sorted['Avg_Moves'], color=colors)
            ax2.set_xlabel('Average Moves per Game')
            ax2.set_title('B) Game Efficiency (Fewer Moves = Better)')
            
            # Add value labels
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}', ha='left', va='center')
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'task4_heuristic_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heuristic comparison chart saved: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating heuristic comparison plot: {e}")
    
    def _plot_tournament_matrix(self, data):
        """Plot tournament results as a matrix."""
        try:
            # Create tournament matrix
            configs = sorted(list(set(data['Config1'].tolist() + data['Config2'].tolist())))
            n_configs = len(configs)
            matrix = np.zeros((n_configs, n_configs))
            
            for _, row in data.iterrows():
                config1 = row['Config1']
                config2 = row['Config2']
                if config1 in configs and config2 in configs:
                    i = configs.index(config1)
                    j = configs.index(config2)
                    # Win rate of config1 vs config2
                    matrix[i][j] = row['P1_Win_Rate']
                    matrix[j][i] = row['P2_Win_Rate']
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            
            if HAS_SEABORN:
                sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                           xticklabels=configs, yticklabels=configs,
                           cbar_kws={'label': 'Win Rate'})
            else:
                # Fallback to matplotlib
                im = plt.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
                plt.colorbar(im, label='Win Rate')
                plt.xticks(range(len(configs)), configs, rotation=45)
                plt.yticks(range(len(configs)), configs)
                
                # Add text annotations
                for i in range(len(configs)):
                    for j in range(len(configs)):
                        plt.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center')
            
            plt.title('Task 5: AI vs AI Tournament Matrix\n(Row vs Column Win Rates)', fontsize=14, fontweight='bold')
            plt.xlabel('Opponent Configuration')
            plt.ylabel('Player Configuration')
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, 'task5_tournament_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Tournament matrix saved: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating tournament matrix plot: {e}")
    
    def _plot_vs_random_performance(self, data):
        """Plot performance vs random agent."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Task 5: AI Performance vs Random Agent', fontsize=16, fontweight='bold')
            
            # Extract config names
            data = data.copy()
            data['Config'] = data['Parameter'].str.replace('_vs_random', '')
            data_sorted = data.sort_values('P2_Win_Rate', ascending=True)
            
            # Win rates
            n_configs = len(data_sorted)
            if HAS_MATPLOTLIB:
                colors = plt.cm.viridis(np.linspace(0, 1, n_configs))
            else:
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'][:n_configs]
                
            bars1 = ax1.barh(data_sorted['Config'], data_sorted['P2_Win_Rate'], color=colors)
            ax1.set_xlabel('Win Rate vs Random')
            ax1.set_title('A) Win Rate Rankings')
            ax1.set_xlim(0, 1)
            
            # Add value labels
            for bar in bars1:
                width = bar.get_width()
                ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            # Efficiency (moves vs win rate)
            scatter = ax2.scatter(data['Avg_Moves'], data['P2_Win_Rate'], 
                                s=100, c=range(len(data)), cmap='viridis', alpha=0.7)
            
            for i, row in data.iterrows():
                ax2.annotate(row['Config'], (row['Avg_Moves'], row['P2_Win_Rate']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Average Moves per Game')
            ax2.set_ylabel('Win Rate vs Random')
            ax2.set_title('B) Efficiency Analysis\n(Top-left = Best: High win rate, fewer moves)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'task5_vs_random_performance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ VS Random performance chart saved: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating vs random performance plot: {e}")
    
    def _create_depth_table(self, data):
        """Create formatted depth analysis table."""
        try:
            data = data.copy()
            data['Depth'] = data['Parameter'].str.extract(r'depth_(\d+)')[0].astype(int)
            data = data.sort_values('Depth')
            
            # Select and format columns
            columns = ['Depth', 'P2_Win_Rate', 'Avg_Moves']
            if 'Avg_Nodes_P2' in data.columns:
                columns.append('Avg_Nodes_P2')
            columns.append('Total_Time')
            
            table_data = data[columns].copy()
            
            # Rename columns
            column_names = ['Depth', 'Win Rate', 'Avg Moves']
            if 'Avg_Nodes_P2' in columns:
                column_names.append('Avg Nodes')
            column_names.append('Time (s)')
            table_data.columns = column_names
            
            # Format for report
            table_data['Win Rate'] = table_data['Win Rate'].apply(lambda x: f"{x:.3f}")
            table_data['Avg Moves'] = table_data['Avg Moves'].apply(lambda x: f"{x:.1f}")
            if 'Avg Nodes' in table_data.columns:
                table_data['Avg Nodes'] = table_data['Avg Nodes'].apply(lambda x: f"{x:.0f}")
            table_data['Time (s)'] = table_data['Time (s)'].apply(lambda x: f"{x:.1f}")
            
            print("\n📋 DEPTH ANALYSIS TABLE (for report):")
            print(table_data.to_string(index=False))
            
            # Save to CSV
            save_path = os.path.join(self.results_dir, 'report_depth_table.csv')
            table_data.to_csv(save_path, index=False)
            print(f"✅ Depth table saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error creating depth table: {e}")
    
    def _create_timeout_table(self, data):
        """Create formatted timeout analysis table."""
        try:
            data = data.copy()
            data['Timeout'] = data['Parameter'].str.extract(r'timeout_(\d+\.?\d*)')[0].astype(float)
            data = data.sort_values('Timeout')
            
            table_data = data[['Timeout', 'P1_Win_Rate', 'P2_Win_Rate', 'Avg_Duration']].copy()
            table_data.columns = ['Timeout (s)', 'Balanced Win Rate', 'Aggressive Win Rate', 'Avg Duration (s)']
            
            # Format for report
            table_data['Balanced Win Rate'] = table_data['Balanced Win Rate'].apply(lambda x: f"{x:.3f}")
            table_data['Aggressive Win Rate'] = table_data['Aggressive Win Rate'].apply(lambda x: f"{x:.3f}")
            table_data['Avg Duration (s)'] = table_data['Avg Duration (s)'].apply(lambda x: f"{x:.1f}")
            
            print("\n📋 TIMEOUT ANALYSIS TABLE (for report):")
            print(table_data.to_string(index=False))
            
            # Save to CSV
            save_path = os.path.join(self.results_dir, 'report_timeout_table.csv')
            table_data.to_csv(save_path, index=False)
            print(f"✅ Timeout table saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error creating timeout table: {e}")
    
    def _create_heuristic_table(self, data):
        """Create formatted heuristic comparison table."""
        try:
            data_sorted = data.sort_values('P2_Win_Rate', ascending=False)
            
            table_data = data_sorted[['Parameter', 'P2_Win_Rate', 'Avg_Moves', 'Avg_Duration']].copy()
            table_data.columns = ['Heuristic', 'Win Rate', 'Avg Moves', 'Avg Duration (s)']
            
            # Format for report
            table_data['Win Rate'] = table_data['Win Rate'].apply(lambda x: f"{x:.3f}")
            table_data['Avg Moves'] = table_data['Avg Moves'].apply(lambda x: f"{x:.1f}")
            table_data['Avg Duration (s)'] = table_data['Avg Duration (s)'].apply(lambda x: f"{x:.1f}")
            
            # Add performance ranking
            table_data['Rank'] = range(1, len(table_data) + 1)
            table_data = table_data[['Rank', 'Heuristic', 'Win Rate', 'Avg Moves', 'Avg Duration (s)']]
            
            print("\n📋 HEURISTIC PERFORMANCE TABLE (for report):")
            print(table_data.to_string(index=False))
            
            # Save to CSV
            save_path = os.path.join(self.results_dir, 'report_heuristic_table.csv')
            table_data.to_csv(save_path, index=False)
            print(f"✅ Heuristic table saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error creating heuristic table: {e}")
    
    def _create_tournament_table(self, data):
        """Create formatted tournament results table."""
        try:
            # Create a summary of tournament results
            tournament_summary = []
            
            for _, row in data.iterrows():
                config1, config2 = row['Config1'], row['Config2']
                p1_wins, p2_wins = row['P1_Wins'], row['P2_Wins']
                winner = config1 if p1_wins > p2_wins else config2 if p2_wins > p1_wins else "Draw"
                score = f"{p1_wins}-{p2_wins}"
                
                tournament_summary.append({
                    'Matchup': f"{config1} vs {config2}",
                    'Winner': winner,
                    'Score': score,
                    'Win_Rate': max(row['P1_Win_Rate'], row['P2_Win_Rate'])
                })
            
            if tournament_summary:
                tournament_df = pd.DataFrame(tournament_summary)
                tournament_df['Win_Rate'] = tournament_df['Win_Rate'].apply(lambda x: f"{x:.3f}")
                
                print("\n📋 TOURNAMENT RESULTS TABLE (for report):")
                print(tournament_df.to_string(index=False))
                
                # Save to CSV
                save_path = os.path.join(self.results_dir, 'report_tournament_table.csv')
                tournament_df.to_csv(save_path, index=False)
                print(f"✅ Tournament table saved: {save_path}")
                
        except Exception as e:
            print(f"❌ Error creating tournament table: {e}")
    
    def _create_performance_table(self, data):
        """Create AI performance summary table."""
        try:
            data = data.copy()
            data['Config'] = data['Parameter'].str.replace('_vs_random', '')
            data_sorted = data.sort_values('P2_Win_Rate', ascending=False)
            
            table_data = data_sorted[['Config', 'P2_Win_Rate', 'Avg_Moves', 'Avg_Duration']].copy()
            table_data.columns = ['Configuration', 'Win Rate', 'Avg Moves', 'Avg Duration (s)']
            
            # Format for report
            table_data['Win Rate'] = table_data['Win Rate'].apply(lambda x: f"{x:.3f}")
            table_data['Avg Moves'] = table_data['Avg Moves'].apply(lambda x: f"{x:.1f}")
            table_data['Avg Duration (s)'] = table_data['Avg Duration (s)'].apply(lambda x: f"{x:.1f}")
            
            # Add performance ranking
            table_data['Rank'] = range(1, len(table_data) + 1)
            table_data = table_data[['Rank', 'Configuration', 'Win Rate', 'Avg Moves', 'Avg Duration (s)']]
            
            print("\n📋 AI CONFIGURATION PERFORMANCE TABLE (for report):")
            print(table_data.to_string(index=False))
            
            # Save to CSV
            save_path = os.path.join(self.results_dir, 'report_performance_table.csv')
            table_data.to_csv(save_path, index=False)
            print(f"✅ Performance table saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error creating performance table: {e}")
    
    def _create_overall_rankings(self):
        """Create overall AI configuration rankings."""
        if not HAS_PANDAS or self.task5_data is None:
            return
        
        try:
            vs_random_data = self.task5_data[self.task5_data['Experiment_Type'] == 'vs_random']
            tournament_data = self.task5_data[self.task5_data['Experiment_Type'] == 'tournament']
            
            if vs_random_data.empty:
                print("ℹ️  No vs_random data available for rankings")
                return
            
            # Calculate overall scores
            scores = {}
            
            # Score from vs random performance
            for _, row in vs_random_data.iterrows():
                config = row['Parameter'].replace('_vs_random', '')
                scores[config] = row['P2_Win_Rate'] * 100  # Base score from win rate
            
            # Bonus points from tournament wins
            for _, row in tournament_data.iterrows():
                config1, config2 = row['Config1'], row['Config2']
                if config1 in scores and config2 in scores:
                    if row['P1_Win_Rate'] > row['P2_Win_Rate']:
                        scores[config1] += 5  # Bonus for tournament win
                    elif row['P2_Win_Rate'] > row['P1_Win_Rate']:
                        scores[config2] += 5
            
            # Create rankings
            ranked_configs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            print("\n🏆 OVERALL AI CONFIGURATION RANKINGS:")
            print("Rank | Configuration | Score  | Performance")
            print("-" * 45)
            for i, (config, score) in enumerate(ranked_configs, 1):
                performance = "Excellent" if score > 90 else "Good" if score > 80 else "Fair" if score > 70 else "Poor"
                print(f" {i:2d}  | {config:12} | {score:5.1f}  | {performance}")
            
            # Save rankings to CSV
            rankings_df = pd.DataFrame(ranked_configs, columns=['Configuration', 'Score'])
            rankings_df['Rank'] = range(1, len(rankings_df) + 1)
            rankings_df['Performance'] = rankings_df['Score'].apply(
                lambda x: "Excellent" if x > 90 else "Good" if x > 80 else "Fair" if x > 70 else "Poor"
            )
            rankings_df = rankings_df[['Rank', 'Configuration', 'Score', 'Performance']]
            
            save_path = os.path.join(self.results_dir, 'report_overall_rankings.csv')
            rankings_df.to_csv(save_path, index=False)
            print(f"✅ Overall rankings saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error creating overall rankings: {e}")
    
    def generate_report_summary(self):
        """Generate a comprehensive summary for the assignment report."""
        print("\n" + "="*80)
        print("ASSIGNMENT REPORT SUMMARY")
        print("="*80)
        
        try:
            if self.task4_data is not None:
                print("\n📊 TASK 4 FINDINGS:")
                
                # Depth Analysis
                depth_data = self.task4_data[self.task4_data['Experiment_Type'] == 'depth_analysis']
                if not depth_data.empty:
                    best_depth = depth_data.loc[depth_data['P2_Win_Rate'].idxmax()]
                    depth_val = best_depth['Parameter'].split('_')[1]
                    print("1. OPTIMAL SEARCH DEPTH:")
                    print(f"   - Best performing depth: {depth_val}")
                    print(f"   - Win rate: {best_depth['P2_Win_Rate']:.3f}")
                    print(f"   - Trade-off: Higher depth improves performance but increases computation time")
                
                # Heuristic Analysis
                heuristic_data = self.task4_data[self.task4_data['Experiment_Type'] == 'heuristic_analysis']
                if not heuristic_data.empty:
                    best_heuristic = heuristic_data.loc[heuristic_data['P2_Win_Rate'].idxmax()]
                    print("\n2. HEURISTIC EFFECTIVENESS:")
                    print(f"   - Best heuristic: {best_heuristic['Parameter']}")
                    print(f"   - Win rate: {best_heuristic['P2_Win_Rate']:.3f}")
            
            if self.task5_data is not None:
                print("\n🏆 TASK 5 FINDINGS:")
                vs_random = self.task5_data[self.task5_data['Experiment_Type'] == 'vs_random']
                if not vs_random.empty:
                    best_ai = vs_random.loc[vs_random['P2_Win_Rate'].idxmax()]
                    config_name = best_ai['Parameter'].replace('_vs_random', '')
                    print(f"   - Best overall AI: {config_name}")
                    print(f"   - Win rate vs random: {best_ai['P2_Win_Rate']:.3f}")
            
            print(f"\n📁 All charts and tables saved in: {self.results_dir}/")
            print("📋 Import CSV files into your report for formatted tables")
            print("🖼️  Include PNG images in your report for visualizations")
            
        except Exception as e:
            print(f"❌ Error generating report summary: {e}")


def main():
    """Main analysis runner."""
    print("📊 RESULTS ANALYZER AND REPORT GENERATOR")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    if not HAS_PANDAS:
        missing_deps.append("pandas")
    if not HAS_MATPLOTLIB:
        missing_deps.append("matplotlib")
    if not HAS_SEABORN:
        missing_deps.append("seaborn")
    
    if missing_deps:
        print("⚠️  Some optional libraries are missing:")
        print(f"   Missing: {', '.join(missing_deps)}")
        print(f"   Install with: pip install {' '.join(missing_deps)}")
        print("   Analysis will continue with limited functionality.\n")
    
    analyzer = ResultsAnalyzer()
    
    while True:
        print(f"\nAnalysis Options:")
        print("1. Load Latest Results")
        print("2. Generate Task 4 Analysis (Charts & Tables)")
        print("3. Generate Task 5 Analysis (Charts & Tables)")
        print("4. Generate Full Report Summary")
        print("5. Generate All Analysis")
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice (0-5): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice == "0":
            break
        
        elif choice == "1":
            analyzer.load_latest_results()
        
        elif choice == "2":
            if analyzer.task4_data is None:
                print("Loading results first...")
                if not analyzer.load_latest_results():
                    continue
            analyzer.generate_task4_analysis()
        
        elif choice == "3":
            if analyzer.task5_data is None:
                print("Loading results first...")
                if not analyzer.load_latest_results():
                    continue
            analyzer.generate_task5_analysis()
        
        elif choice == "4":
            if analyzer.task4_data is None or analyzer.task5_data is None:
                print("Loading results first...")
                if not analyzer.load_latest_results():
                    continue
            analyzer.generate_report_summary()
        
        elif choice == "5":
            print("Loading results and generating all analysis...")
            if analyzer.load_latest_results():
                analyzer.generate_task4_analysis()
                analyzer.generate_task5_analysis()
                analyzer.generate_report_summary()
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()