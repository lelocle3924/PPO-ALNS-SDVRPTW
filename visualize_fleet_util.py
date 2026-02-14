import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION: PATHS TO SPECIFIC SUMMARY FILES ---
FILES = {
    "PPO_Jan13": "25240114_0317/route_summary_ppo_1-13-2023.csv",
    "OR_Jan13": "2524-20260113T180717Z-1-001/2524/route_summary_ortools_2023-01-13.csv",
    "PPO_Mar16": "25240114_0317/route_summary_ppo_3-16-2023.csv",
    "OR_Mar16": "2524-20260113T180717Z-1-001/2524/route_summary_ortools_2023-03-16.csv"
}

def load_data():
    dfs = {}
    for key, path in FILES.items():
        try:
            df = pd.read_csv(path)
            df['Scenario'] = "Normal (Jan 13)" if "Jan13" in key else "Peak (Mar 16)"
            df['Method'] = "PPO-ALNS" if "PPO" in key else "OR-Tools"
            
            if 'VehicleType' in df.columns:
                df['Vehicle_Category'] = df['VehicleType'].apply(lambda x: 'MC' if 'MC' in x else ('AUV' if 'AUV' in x else ('6w' if '6w' in x else '4w')))
            
            dfs[key] = df
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None
    return pd.concat(dfs.values(), ignore_index=True)

def plot_fleet_mix(df_all):
    """Generates Stacked Bar Chart for Fleet Composition"""
    fleet_counts = df_all.groupby(['Scenario', 'Method', 'Vehicle_Category']).size().reset_index(name='Count')
    
    pivot_df = fleet_counts.pivot_table(index=['Scenario', 'Method'], columns='Vehicle_Category', values='Count', fill_value=0)
    
    pivot_df = pivot_df.reindex([('Normal (Jan 13)', 'OR-Tools'), ('Normal (Jan 13)', 'PPO-ALNS'), 
                                 ('Peak (Mar 16)', 'OR-Tools'), ('Peak (Mar 16)', 'PPO-ALNS')])

    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title('Fleet Composition Strategy: PPO vs. OR-Tools', fontsize=14)
    plt.xlabel('Scenario & Method', fontsize=12)
    plt.ylabel('Number of Vehicles', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Vehicle Type')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for c in ax.containers:
        ax.bar_label(c, label_type='center', color='white', weight='bold')
        
    plt.tight_layout()
    plt.savefig("fleet_composition.png", dpi=300)
    print(">> Saved 'fleet_composition.png'")

def plot_utilization_boxplot(df_all):
    """Generates Box Plot for Capacity Utilization"""
    plt.figure(figsize=(10, 6))
    
    util_col = 'Util_KGM_%'
    
    sns.boxplot(data=df_all, x='Scenario', y=util_col, hue='Method', 
                palette={'OR-Tools': '#4c72b0', 'PPO-ALNS': '#dd8452'}, width=0.6)
    
    plt.title('Capacity Utilization Efficiency (Bin Packing)', fontsize=14)
    plt.ylabel('Weight Utilization (%)', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Max Capacity')
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("utilization_boxplot.png", dpi=300)
    print(">> Saved 'utilization_boxplot.png'")

if __name__ == "__main__":
    combined_df = load_data()
    if combined_df is not None:
        plot_fleet_mix(combined_df)
        plot_utilization_boxplot(combined_df)
    else:
        print("Could not load data. Please check file paths.")