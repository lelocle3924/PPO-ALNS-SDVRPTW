import pandas as pd
import numpy as np
import os
from typing import List
from core.data_structures import ProblemData, VehicleType
from core.distance_matrix import DistanceMatrixCalculator
from config import PathConfig
class RealDataLoader:
    def __init__(self):
        self.fleet: List[VehicleType] = []
        self.vehicle_map = {}

    def _normalize_id(self, val) -> str:
        if pd.isna(val): return ""
        s = str(val).strip()
        if s.endswith('.0'): s = s[:-2]
        return s

    def _parse_time(self, time_str):
        if pd.isna(time_str): return 0
        try:
            if isinstance(time_str, str):
                parts = time_str.split(':')
                return int(parts[0]) * 60 + int(parts[1])
        except: pass
        return 0

    def _parse_allowed_trucks(self, allowed_str: str) -> List[int]:
        if pd.isna(allowed_str): return [v.type_id for v in self.fleet]
        
        clean_str = str(allowed_str).replace('{', '').replace('}', '').replace(' ', '').replace('"', '')
        types = clean_str.split(',')
        
        allowed_ids = []
        for t in types:
            if t in self.vehicle_map:
                allowed_ids.append(self.vehicle_map[t])
        
        return allowed_ids if allowed_ids else [v.type_id for v in self.fleet]

    def _load_fleet_from_csv(self, truck_csv_path: str):
        print(f"  > Loading Fleet Config from {truck_csv_path}...")
        try:
            df_truck = pd.read_csv(truck_csv_path)
            df_truck.columns = df_truck.columns.str.strip()
            
            self.fleet = []
            for idx, row in df_truck.iterrows():
                v = VehicleType(
                    type_id=idx,
                    name=str(row['TruckName']).strip(),
                    capacity_kg=float(row['CapacityKg']),
                    capacity_cbm=float(row['CapacityCbm']),
                    speed_kmh=float(row['AverageSpeedKmH']),
                    fixed_cost=float(row['FixedCost']),
                    cost_per_km=float(row['CostPerKm']),
                    cost_per_hour=float(row['CostPerHour']),
                    count=50 
                )
                self.fleet.append(v)
            
            self.vehicle_map = {v.name: v.type_id for v in self.fleet}
            
        except Exception as e:
            print(f"[CRITICAL] Error loading Truck Master: {e}")
            raise e

    def load_day_data(self, order_csv_path: str, truck_csv_path: str, override_depot_id: str = None) -> ProblemData:
        print(f"--- Loading Data Pipeline ---")
        
        # 1. LOAD FLEET
        self._load_fleet_from_csv(truck_csv_path)
        
        # 2. LOAD & AGGREGATE ORDERS
        df_orders_raw = pd.read_csv(order_csv_path)
        df_orders_raw['KGM'] = df_orders_raw['KGM'].fillna(0)
        df_orders_raw['CBM'] = df_orders_raw['CBM'].fillna(0)
        
        agg_rules = {
            'KGM': 'sum', 'CBM': 'sum', 
            'CusLat': 'first', 'CusLong': 'first',
            'Beginning1': 'first', 'Ending1': 'first',
            'DwellTime': 'first', 'AllowedTrucks': 'first',
            'Depot': 'first', 'DepotLat': 'first', 'DepotLong': 'first'
        }
        df_orders = df_orders_raw.groupby('Customer', as_index=False).agg(agg_rules)
        
        raw_depot_id = df_orders.iloc[0]['Depot']
        depot_id_from_data = self._normalize_id(raw_depot_id)
        
        filename = os.path.basename(order_csv_path)
        if override_depot_id:
            depot_id = str(override_depot_id)
            print(f"  > [CONFIG] Using Override Cache ID: {depot_id}")
        else:
            filename = os.path.basename(order_csv_path)
            try:
                depot_id = filename.replace('Split_TransportOrder_', '').replace('.csv', '')
                if 'orders_' in depot_id:
                     depot_id = depot_id.replace('orders_', '')
            except:
                depot_id = depot_id_from_data
        
        # NOTE: node_ids here are STRINGS
        node_ids = [depot_id_from_data] + df_orders['Customer'].map(self._normalize_id).tolist()
        num_nodes = len(node_ids)

        # 3. GET DISTANCE MATRIX (CACHE vs CALCULATE)
        dist_matrix_meters = None
        
        cache_dir = PathConfig.DISTANCE_TIME_PATH
        cache_file_name = f"distance_matrix_meters{depot_id}.csv"
        cache_path = os.path.join(cache_dir, cache_file_name)
        
        # --- CACHE CHECK ---
        if os.path.exists(cache_path):
            print(f"  > [CACHE HIT] Found existing matrix: {cache_path}")
            try:
                # Load CSV (Pandas might infer Int64 for index if IDs look like numbers)
                df_cache = pd.read_csv(cache_path, index_col=0)
                
                # [CRITICAL FIX] Force Index and Columns to String to match node_ids
                df_cache.index = df_cache.index.astype(str)
                df_cache.columns = df_cache.columns.astype(str)
                
                # Reindex
                df_aligned = df_cache.reindex(index=node_ids, columns=node_ids, fill_value=1e9)
                
                # Fill Diagonal = 0
                np.fill_diagonal(df_aligned.values, 0)
                
                dist_matrix_meters = df_aligned.to_numpy()
                
                # Safety Check: Nếu quá 50% là vô cực -> Có thể mismatch -> Force Recalculate
                if np.mean(dist_matrix_meters >= 1e9) > 0.5:
                    print("    [!] Warning: Cache seems mismatched (too many Infs). Forcing OSRM.")
                    dist_matrix_meters = None
                else:
                    print(f"    -> Loaded Matrix Shape: {dist_matrix_meters.shape}")
                
            except Exception as e:
                print(f"    [!] Error reading cache: {e}. Fallback to OSRM.")
                dist_matrix_meters = None
        else:
            print(f"  > [CACHE MISS] File {cache_file_name} not found.")

        # --- OSRM FALLBACK ---
        if dist_matrix_meters is None:
            print("  > Invoking OSRM Calculation...")
            calculator = DistanceMatrixCalculator(order_csv_path, truck_csv_path)
            
            # Tính toán
            dist_matrix_meters, _ = calculator.calculate_matrices()
            
            # Save to Cache
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # Khi save, đảm bảo index là node_ids hiện tại để khớp cho lần sau
            df_save = pd.DataFrame(dist_matrix_meters, index=node_ids, columns=node_ids)
            print(f"  > Saving new matrix to cache: {cache_path}")
            df_save.to_csv(cache_path)

        # 4. BUILD SUPER TIME MATRIX
        num_vehicle_types = len(self.fleet)
        super_time_matrix = np.zeros((num_vehicle_types, num_nodes, num_nodes))

        for v in self.fleet:
            speed_mpm = v.speed_kmh * 1000.0 / 60.0 
            safe_speed = max(0.1, speed_mpm)
            time_matrix_v = dist_matrix_meters / safe_speed
            super_time_matrix[v.type_id] = time_matrix_v
            
        # 5. Fill Node Attributes
        coords = np.zeros((num_nodes, 2))
        demands_kg = np.zeros(num_nodes)
        demands_cbm = np.zeros(num_nodes)
        time_windows = np.zeros((num_nodes, 2))
        service_times = np.zeros(num_nodes)
        allowed_vehicles = []

        # Depot
        depot_row = df_orders.iloc[0]
        coords[0] = [depot_row['DepotLat'], depot_row['DepotLong']]
        time_windows[0] = [6 * 60, 24 * 60]
        allowed_vehicles.append([v.type_id for v in self.fleet])

        # Customers
        for i, row in df_orders.iterrows():
            idx = i + 1
            coords[idx] = [row['CusLat'], row['CusLong']]
            demands_kg[idx] = float(row['KGM'])
            demands_cbm[idx] = float(row['CBM'])
            
            start = self._parse_time(row['Beginning1'])
            end = self._parse_time(row['Ending1'])
            if end <= start: end = 18 * 60 
            time_windows[idx] = [start, end]
            
            dwell = float(row['DwellTime']) if not pd.isna(row['DwellTime']) else 0.5
            service_times[idx] = dwell * 60
            
            allowed_vehicles.append(self._parse_allowed_trucks(row['AllowedTrucks']))
        
        return ProblemData(
            dist_matrix=dist_matrix_meters,
            super_time_matrix=super_time_matrix,
            node_ids=node_ids,
            coords=coords,
            demands_kg=demands_kg,
            demands_cbm=demands_cbm,
            time_windows=time_windows,
            service_times=service_times,
            allowed_vehicles=allowed_vehicles,
            vehicle_types=self.fleet
        )