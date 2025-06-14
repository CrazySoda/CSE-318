#!/usr/bin/env python3
"""
Results Analyzer and Report Generator
Analyzes experiment results and generates tables/charts for assignment report
"""

import json
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class ResultsAnalyzer:
    """Analyzes experiment results and generates visualizations."""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = results_dir
        self.task4_data = None
        self.task5_data = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_latest_results(self):
        """Load the most recent experiment results."""
        try:
            # Find latest files
            task4_files = [f for f in os.listdir(self.results_dir) if f.startswith('task4_summary_')]
            task5_files = [f for f in os.listdir(self.results_dir) if f.startswith('task5_summary_')]
            
            if task4_files:
                latest_task4 = max(task4_files)
                self.task4_data = pd.read_csv(os.path.join(self.results_dir, latest_task4))
                print(f"✅ Loaded Task 4 data: {latest_task4}")
            
            if task5_files:
                latest_task5 = max(task5_files)
                self.task5_data = pd.read_csv(os.path.join(self.results_dir, latest_task5))
                print(f"✅ Loaded Task 5 data: {latest_task5}")
                
        except Exception as e:
            print(f"❌ Error loading results: {e}")
    
    def generate_task4_analysis(self):
        """Generate Task 4 analysis and visualizations."""
        if self.task4_data is None:
            print("No Task 4 data available!")
            return
        
        print("\n📊 GENERATING TASK 4 ANALYSIS...")
        
        # 1. Depth Analysis Chart
        depth_data = self.task4_data[self.task4_data['Experiment_Type'] == 'depth_analysis']
        if not depth_data.empty:
            self._plot_depth_analysis(depth_data)
            self._create_depth_table(depth_data)
        
        # 2. Timeout Analysis Chart
        timeout_data = self.task4_data[self.task4_data['Experiment_Type'] == 'timeout_analysis']
        if not timeout_data.empty:
            self._plot_timeout_analysis(timeout_data)
        
        # 3. Heuristic Comparison
        heuristic_data = self.task4_data[self.task4_data['Experiment_Type'] == 'heuristic_analysis']
        if not heuristic_data.empty:
            self._plot_heuristic_comparison(heuristic_data)
            self._create_heuristic_table(heuristic_data)
    
    def generate_task5_analysis(self):
        """Generate Task 5 analysis and visualizations."""
        if self.task5_data is None:
            print("No Task 5 data available!")
            return
        
        print("\n📊 GENERATING TASK 5 ANALYSIS...")
        
        # 1. Tournament Matrix
        tournament_data = self.task5_data[self.task5_data['Experiment_Type'] == 'tournament']
        if not tournament_data.empty:
            self._plot_tournament_matrix(tournament_data)
        
        # 2. Performance vs Random
        vs_random_data = self.task5_data[self.task5_data['Experiment_Type'] == 'vs_random']
        if not vs_random_data.empty:
            self._plot_vs_random_performance(vs_random_data)
            self._create_performance_table(vs_random_data)
        
        # 3. Overall Rankings
        self._create_overall_rankings()
    
    def _plot_depth_analysis(self, data):
        """Plot depth vs performance analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Task 4: Search Depth Analysis (vs Random Agent)', fontsize=16, fontweight='bold')
        
        # Extract depth from parameter names
        data['Depth'] = data['Parameter'].str.extract('(\d+)').astype(int)
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
        
        # Nodes Explored vs Depth
        ax3.semilogy(data['Depth'], data['Avg_Nodes_P2'], 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Search Depth')
        ax3.set_ylabel('Average Nodes Explored (log scale)')
        ax3.set_title('C) Computational Cost vs Search Depth')
        ax3.grid(True, alpha=0.3)
        
        # Performance Efficiency (Win Rate / Computation)
        efficiency = data['P2_Win_Rate'] / (data['Avg_Nodes_P2'] / 1000)  # Normalize nodes
        ax4.plot(data['Depth'], efficiency, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Search Depth')
        ax4.set_ylabel('Efficiency (Win Rate / Computation)')
        ax4.set_title('D) Performance Efficiency vs Depth')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'task4_depth_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_timeout_analysis(self, data):
        """Plot timeout analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Task 4: Timeout Analysis (Balanced vs Aggressive)', fontsize=16, fontweight='bold')
        
        # Extract timeout values
        data['Timeout'] = data['Parameter'].str.extract('(\d+\.?\d*)').astype(float)
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
        plt.savefig(os.path.join(self.results_dir, 'task4_timeout_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_heuristic_comparison(self, data):
        """Plot heuristic comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Task 4: Heuristic Effectiveness Analysis', fontsize=16, fontweight='bold')
        
        # Sort by win rate
        data_sorted = data.sort_values('P2_Win_Rate', ascending=True)
        
        # Horizontal bar chart of win rates
        colors = ['lightcoral', 'gold', 'lightgreen', 'skyblue']
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
        plt.savefig(os.path.join(self.results_dir, 'task4_heuristic_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_tournament_matrix(self, data):
        """Plot tournament results as a matrix."""
        # Create tournament matrix
        configs = ['balanced', 'aggressive', 'defensive', 'tactical', 'strategic', 'fast']
        matrix = np.zeros((len(configs), len(configs)))
        
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
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   xticklabels=configs, yticklabels=configs,
                   cbar_kws={'label': 'Win Rate'})
        plt.title('Task 5: AI vs AI Tournament Matrix\n(Row vs Column Win Rates)', fontsize=14, fontweight='bold')
        plt.xlabel('Opponent Configuration')
        plt.ylabel('Player Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'task5_tournament_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_vs_random_performance(self, data):
        """Plot performance vs random agent."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Task 5: AI Performance vs Random Agent', fontsize=16, fontweight='bold')
        
        # Extract config names
        data['Config'] = data['Parameter'].str.replace('_vs_random', '')
        data_sorted = data.sort_values('P2_Win_Rate', ascending=True)
        
        # Win rates
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_sorted)))
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
        plt.savefig(os.path.join(self.results_dir, 'task5_vs_random_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_depth_table(self, data):
        """Create formatted depth analysis table."""
        data['Depth'] = data['Parameter'].str.extract('(\d+)').astype(int)
        data = data.sort_values('Depth')
        
        table_data = data[['Depth', 'P2_Win_Rate', 'Avg_Moves', 'Avg_Nodes_P2', 'Total_Time']].copy()
        table_data.columns = ['Depth', 'Win Rate', 'Avg Moves', 'Avg Nodes', 'Time (s)']
        
        # Format for report
        table_data['Win Rate'] = table_data['Win Rate'].apply(lambda x: f"{x:.3f}")
        table_data['Avg Moves'] = table_data['Avg Moves'].apply(lambda x: f"{x:.1f}")
        table_data['Avg Nodes'] = table_data['Avg Nodes'].apply(lambda x: f"{x:.0f}")
        table_data['Time (s)'] = table_data['Time (s)'].apply(lambda x: f"{x:.1f}")
        
        print("\n📋 DEPTH ANALYSIS TABLE (for report):")
        print(table_data.to_string(index=False))
        
        # Save to CSV
        table_data.to_csv(os.path.join(self.results_dir, 'report_depth_table.csv'), index=False)
    
    def _create_heuristic_table(self, data):
        """Create formatted heuristic comparison table."""
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
        table_data.to_csv(os.path.join(self.results_dir, 'report_heuristic_table.csv'), index=False)
    
    def _create_performance_table(self, data):
        """Create AI performance summary table."""
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
        table_data.to_csv(os.path.join(self.results_dir, 'report_performance_table.csv'), index=False)
    
    def _create_overall_rankings(self):
        """Create overall AI configuration rankings."""
        if self.task5_data is None:
            return
        
        vs_random_data = self.task5_data[self.task5_data['Experiment_Type'] == 'vs_random']
        tournament_data = self.task5_data[self.task5_data['Experiment_Type'] == 'tournament']
        
        if vs_random_data.empty:
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
    
    def generate_report_summary(self):
        """Generate a comprehensive summary for the assignment report."""
        print("\n" + "="*80)
        print("ASSIGNMENT REPORT SUMMARY")
        print("="*80)
        
        if self.task4_data is not None:
            print("\n📊 TASK 4 FINDINGS:")
            print("1. OPTIMAL SEARCH DEPTH:")
            depth_data = self.task4_data[self.task4_data['Experiment_Type'] == 'depth_analysis']
            if not depth_data.empty:
                best_depth = depth_data.loc[depth_data['P2_Win_Rate'].idxmax()]
                print(f"   - Best performing depth: {best_depth['Parameter'].split('_')[1]}")
                print(f"   - Win rate: {best_depth['P2_Win_Rate']:.3f}")
                print(f"   - Trade-off: Higher depth improves performance but increases computation time")
            
            print("\n2. HEURISTIC EFFECTIVENESS:")
            heuristic_data = self.task4_data[self.task4_data['Experiment_Type'] == 'heuristic_analysis']
            if not heuristic_data.empty:
                best_heuristic = heuristic_data.loc[heuristic_data['P2_Win_Rate'].idxmax()]
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


def main():
    """Main analysis runner."""
    print("📊 RESULTS ANALYZER AND REPORT GENERATOR")
    print("=" * 50)
    
    analyzer = ResultsAnalyzer()
    
    # Check if pandas and matplotlib are available
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("❌ Required libraries not installed!")
        print("Please install: pip install pandas matplotlib seaborn")
        return
    
    while True:
        print(f"\nAnalysis Options:")
        print("1. Load Latest Results")
        print("2. Generate Task 4 Analysis (Charts & Tables)")
        print("3. Generate Task 5 Analysis (Charts & Tables)")
        print("4. Generate Full Report Summary")
        print("5. Generate All Analysis")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == "0":
            break
        
        elif choice == "1":
            analyzer.load_latest_results()
        
        elif choice == "2":
            if analyzer.task4_data is None:
                analyzer.load_latest_results()
            analyzer.generate_task4_analysis()
        
        elif choice == "3":
            if analyzer.task5_data is None:
                analyzer.load_latest_results()
            analyzer.generate_task5_analysis()
        
        elif choice == "4":
            if analyzer.task4_data is None or analyzer.task5_data is None:
                analyzer.load_latest_results()
            analyzer.generate_report_summary()
        
        elif choice == "5":
            analyzer.load_latest_results()
            analyzer.generate_task4_analysis()
            analyzer.generate_task5_analysis()
            analyzer.generate_report_summary()
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()