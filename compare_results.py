import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
PPO_RESULTS_DIR = "results-PPO"  
OR_TOOLS_RESULTS_DIR = "results-ORTools" 

def parse_date_from_filename(filename):
    match_iso = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match_iso:
        try:
            return datetime.strptime(match_iso.group(0), '%Y-%m-%d')
        except ValueError: pass

    match_mdy = re.search(r'(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)', filename)
    if match_mdy:
        try:
            date_str = match_mdy.group(0)
            return datetime.strptime(date_str, '%m-%d-%Y')
        except ValueError: pass

    return None

def load_summaries(result_dir, method_name):
    summary_data = []
    
    files = glob.glob(os.path.join(result_dir, "*.csv"))
    print(f"Scanning {result_dir}... Found {len(files)} files.")
    
    for f in files:
        filename = os.path.basename(f)
        
        if "route_summary" not in filename:
            continue 

        try:
            date_obj = parse_date_from_filename(filename)
            if not date_obj:
                print(f"  [Warning] Could not extract date from: {filename}")
                continue
            
            df = pd.read_csv(f)
            
            if 'Total_Trip_Cost' in df.columns:
                total_cost = df['Total_Trip_Cost'].sum()
            elif 'Route_Total_Cost' in df.columns:
                total_cost = df['Route_Total_Cost'].sum()
            else:
                print(f"  [Error] No cost column found in {filename}")
                continue

            vehicle_count = len(df)
            
            total_orders = 0
            if 'Num_Stops' in df.columns:
                total_orders = df['Num_Stops'].sum()
            
            summary_data.append({
                'Date': date_obj,
                f'Cost_{method_name}': total_cost,
                f'Fleet_{method_name}': vehicle_count,
                f'Orders_{method_name}': total_orders
            })
        except Exception as e:
            print(f"  [Error] Reading {f}: {e}")
            
    return pd.DataFrame(summary_data)

def main():
    # 1. Load Data
    print("Loading PPO Results...")
    df_ppo = load_summaries(PPO_RESULTS_DIR, "PPO")
    if df_ppo.empty:
        print("Error: No PPO data loaded. Check filenames.")
        return

    print("Loading OR-Tools Results...")
    df_or = load_summaries(OR_TOOLS_RESULTS_DIR, "OR")
    if df_or.empty:
        print("Error: No OR-Tools data loaded. Check filenames.")
        return
    
    # 2. Merge on Date (both are datetime objects now)
    df_merged = pd.merge(df_ppo, df_or, on='Date', how='inner')
    
    if df_merged.empty:
        print("Error: No matching dates found between PPO and OR-Tools results.")
        print("PPO Dates:", df_ppo['Date'].dt.strftime('%Y-%m-%d').unique())
        print("OR Dates:", df_or['Date'].dt.strftime('%Y-%m-%d').unique())
        return

    # 3. Calculate Stats
    df_merged['Gap_Percent'] = ((df_merged['Cost_PPO'] - df_merged['Cost_OR']) / df_merged['Cost_OR']) * 100
    df_merged['Fleet_Diff'] = df_merged['Fleet_PPO'] - df_merged['Fleet_OR']
    
    # Sort by date
    df_merged = df_merged.sort_values('Date')
    
    # 4. Print Output
    output_cols = ['Date', 'Orders_OR', 'Fleet_OR', 'Fleet_PPO', 'Cost_OR', 'Cost_PPO', 'Gap_Percent']
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*80)
    
    print_df = df_merged.copy()
    print_df['Date'] = print_df['Date'].dt.strftime('%Y-%m-%d')
    print(print_df[output_cols].to_string(index=False, float_format="%.2f"))
    
    # Calculate Average Gap
    avg_gap = df_merged['Gap_Percent'].mean()
    print("-" * 80)
    print(f"AVERAGE OPTIMALITY GAP: {avg_gap:.2f}%")
    print("=" * 80)

    # Save CSV
    df_merged.to_csv("comparison_stats.csv", index=False)
    print(f"\n>> Table saved to 'comparison_stats.csv'")

    # 5. Generate Chart
    plt.figure(figsize=(14, 7))
    
    dates = df_merged['Date'].dt.strftime('%m-%d')
    x = np.arange(len(dates))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.bar(x - width/2, df_merged['Cost_OR'], width, label='OR-Tools (Benchmark)', color='#4c72b0', alpha=0.8)
    ax1.bar(x + width/2, df_merged['Cost_PPO'], width, label='PPO-ALNS', color='#dd8452', alpha=0.8)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Operational Cost', color='black', fontsize=12)
    ax1.set_title('Cost Comparison & Optimality Gap (14-Day Horizon)', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=45)
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, df_merged['Gap_Percent'], color='red', marker='o', linewidth=2, label='Gap %')
    ax2.set_ylabel('Optimality Gap (%)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    for i, txt in enumerate(df_merged['Gap_Percent']):
        ax2.annotate(f"{txt:.1f}%", (x[i], df_merged['Gap_Percent'].iloc[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', color='red', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig("statistical_comparison.png", dpi=300)
    print(f">> Chart saved to 'statistical_comparison.png'")

if __name__ == "__main__":
    main()