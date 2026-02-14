import pandas as pd
import numpy as np
import os
import shutil
import time 
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from core.data_structures import RvrpState, ProblemData
from core.real_data_loader import RealDataLoader
from core.visualizer import RouteVisualizer
from config import PathConfig, PPOConfig
from ppo.rvrpenv import RVRPEnvironment
from core.logger import LightLogger 
import logging

path_cfg = PathConfig()
ppo_cfg = PPOConfig()

# --- SETUP LOGGER ---
os.makedirs(path_cfg.FINAL_REPORT_DIR, exist_ok=True)
log_file_path = os.path.join(path_cfg.FINAL_REPORT_DIR, "inference.log")

logger = LightLogger("Inference")
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y%m%d-%H:%M:%S")
file_handler.setFormatter(formatter)
logger.logger.addHandler(file_handler)

# --- 1. DATA PRE-PROCESSING ---
def split_orders_by_date(original_csv_path, output_dir):
    logger.info(f">>> [1/4] Splitting Orders from {original_csv_path}...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) 
    os.makedirs(output_dir)
    try:
        df = pd.read_csv(original_csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return []
    date_col = 'Delivery Date' 
    if date_col not in df.columns:
        possible_cols = [c for c in df.columns if 'date' in c.lower()]
        if possible_cols: date_col = possible_cols[0]
    unique_dates = df[date_col].unique()
    generated_files = []
    for d in unique_dates:
        day_df = df[df[date_col] == d].copy()
        agg_rules = {
            'KGM': 'sum', 'CBM': 'sum', 
            'CusLat': 'first', 'CusLong': 'first',
            'Beginning1': 'first', 'Ending1': 'first',
            'DwellTime': 'first', 'AllowedTrucks': 'first',
            'Depot': 'first', 'DepotLat': 'first', 'DepotLong': 'first'
        }
        consolidated_df = day_df.groupby('Customer', as_index=False).agg(agg_rules)
        safe_date = str(d).replace("/", "-").replace(" ", "_")
        fname = os.path.join(output_dir, f"orders_{safe_date}.csv")
        consolidated_df.to_csv(fname, index=False)
        generated_files.append((safe_date, fname))
    return generated_files

# --- 2. MATRIX WARM-UP ---
def warmup_and_get_master_id(full_order_path):
    logger.info(">>> [2/4] Distance Matrix...")
    loader = RealDataLoader()
    problem_data = loader.load_day_data(full_order_path, path_cfg.TRUCK_PATH)
    return problem_data.node_ids[0]

# --- 3. EXPORT FUNCTION ---
def export_rvrp_result_csv(solution: RvrpState, data: ProblemData, date_str: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    detail_rows = []
    depot_open_time = data.time_windows[0][0]
    
    def fmt_time(t): 
        hrs = int(t // 60) % 24
        mns = int(t % 60)
        return f"{hrs:02d}:{mns:02d}"

    for r_idx, route in enumerate(solution.routes):
        if not route.node_sequence: continue
        
        v_type = route.vehicle_type
        route_id = f"R_{date_str}_{r_idx}"
        vehicle_id = f"{v_type.name}_{r_idx}"
        
        first_node = route.node_sequence[0]
        time_to_first = data.get_travel_time(0, first_node, v_type.type_id)
        first_tw_start = data.time_windows[first_node][0]
        optimal_departure = max(depot_open_time, first_tw_start - time_to_first)
        
        current_time = optimal_departure
        prev_node = 0
        total_wait_jit = 0
        total_dist_m = 0
        route_stops_buffer = []
        
        for seq, node_idx in enumerate(route.node_sequence):
            dist_m = data.dist_matrix[prev_node, node_idx]
            travel_m = data.get_travel_time(prev_node, node_idx, v_type.type_id)
            total_dist_m += dist_m
            current_time += travel_m
            arrival_time = current_time
            
            tw_start, tw_end = data.time_windows[node_idx]
            wait_m = max(0, tw_start - current_time)
            total_wait_jit += wait_m
            current_time += wait_m
            
            service_m = data.service_times[node_idx]
            current_time += service_m
            departure_time = current_time
            
            route_stops_buffer.append({
                "RouteID": route_id,
                "VehicleID": vehicle_id,
                "VehicleType": v_type.name,
                "StopOrder": seq + 1,
                "LocationID": data.node_ids[node_idx],
                "ETA": fmt_time(arrival_time),
                "ETD": fmt_time(departure_time),
                "TimeWindow": f"{fmt_time(tw_start)}-{fmt_time(tw_end)}",
                "Demand_KGM": data.demands_kg[node_idx],
                "Segment_Dist_KM": round(dist_m / 1000.0, 2),
                "Wait_Min": round(wait_m, 1)
            })
            prev_node = node_idx

        total_dist_m += data.dist_matrix[prev_node, 0]
        current_time += data.get_travel_time(prev_node, 0, v_type.type_id)
        final_duration = current_time - optimal_departure

        current_load_kg = route.total_load_kg
        for stop in route_stops_buffer:
            stop["Load_On_Board_KG"] = current_load_kg
            current_load_kg -= stop["Demand_KGM"]
            detail_rows.append(stop)

        summary_rows.append({
            "RouteID": route_id,
            "VehicleID": vehicle_id,
            "VehicleType": v_type.name,
            "Departure_From_Depot": fmt_time(optimal_departure),
            "Arrival_Back_Depot": fmt_time(current_time),
            "Num_Stops": len(route.node_sequence),
            "Total_Distance_KM": round(total_dist_m / 1000.0, 2),
            "Total_Duration_Min": round(final_duration, 1),
            "Total_Load_KGM": route.total_load_kg,
            "Capacity_KGM": v_type.capacity_kg,
            "Util_KGM_%": round(route.capacity_utilization * 100, 1),
            "Total_Trip_Cost": round(route.cost, 0)
        })

    # Export CSV
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, f"route_summary_ppo_{date_str}.csv"), index=False, encoding='utf-8-sig')
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(os.path.join(output_dir, f"stop_details_ppo_{date_str}.csv"), index=False, encoding='utf-8-sig')
    
    # Map
    viz = RouteVisualizer(output_dir=output_dir)
    viz.visualize_solution(solution, data, filename=f"map_{date_str}.html")

# --- 4. MAIN ---
def run_inference_pipeline():
    date_files = split_orders_by_date(path_cfg.ORDER_PATH, path_cfg.TEMP_DATA_DIR)
    master_depot_id = warmup_and_get_master_id(path_cfg.ORDER_PATH)
    
    if not os.path.exists(path_cfg.INFERENCE_MODEL_PATH):
        logger.error(" Model not found.")
        return

    model = MaskablePPO.load(path_cfg.INFERENCE_MODEL_PATH, device=ppo_cfg.device)
    def mask_fn(env): return env.valid_action_mask()
    
    for date_str, csv_path in date_files:
        logger.info(f"  Date: {date_str}")
        start_time = time.time()
        try:
            env = RVRPEnvironment(csv_path, path_cfg.TRUCK_PATH, is_test_mode=True, override_depot_id=master_depot_id)
            env = ActionMasker(env, mask_fn)
            obs, _ = env.reset()
            done = False
            while not done:
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            real_env = env.unwrapped
            export_rvrp_result_csv(real_env.best_solution, real_env.problem_data, date_str, path_cfg.FINAL_REPORT_DIR)
            logger.info(f"    Done | Cost: {real_env.best_solution.objective():,.0f}")
        except Exception as e:
            logger.error(f"    Error {date_str}: {e}")

if __name__ == "__main__":
    run_inference_pipeline()