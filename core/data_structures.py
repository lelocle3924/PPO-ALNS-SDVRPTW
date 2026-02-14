import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ==========================================
# PART 1: VEHICLE & FLEET CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class VehicleType:
    """
    Represents a specific type of vehicle in the heterogeneous fleet.
    Frozen for speed and immutability.
    """
    type_id: int            
    name: str               
    capacity_kg: float      
    capacity_cbm: float     
    speed_kmh: float        
    fixed_cost: float       
    cost_per_km: float      
    cost_per_hour: float    
    count: int              

# ==========================================
# PART 2: PROBLEM DATA (Static Context)
# ==========================================
@dataclass
class ProblemData:
    """
    Container for all static data.
    OPTIMIZATION: Static features are pre-calculated to avoid CPU lag in reset/step.
    """
    dist_matrix: np.ndarray 
    super_time_matrix: np.ndarray 

    node_ids: List[str]     
    coords: np.ndarray      # (N, 2) [Lat, Lon]
    demands_kg: np.ndarray  
    demands_cbm: np.ndarray 
    time_windows: np.ndarray 
    service_times: np.ndarray 
    allowed_vehicles: List[List[int]] 

    vehicle_types: List[VehicleType]
    max_route_duration: float = 1440.0 

    _id_to_index: Dict[str, int] = field(default_factory=dict, repr=False)

    # Pre-calculated features for PPO state
    _static_tw_tightness: float = field(init=False, default=0.0)
    _static_spatial_density: float = field(init=False, default=0.0)
    _demands_mean: float = field(init=False, default=0.0)
    _demands_std: float = field(init=False, default=0.0)
    _tw_width_mean: float = field(init=False, default=0.0)

    def __post_init__(self):
        for idx, node_id in enumerate(self.node_ids):
            self._id_to_index[str(node_id)] = idx
            
        # --- [OPTIMIZATION] PRE-CALCULATE STATIC FEATURES ONCE ---
        self._calculate_static_features()

    def _calculate_static_features(self):
        widths = self.time_windows[:, 1] - self.time_windows[:, 0]
        safe_widths = np.maximum(widths, 1.0)
        
        cust_service = self.service_times[1:]
        cust_widths = safe_widths[1:]
        
        if len(cust_widths) > 0:
            self._static_tw_tightness = float(np.mean(cust_service / cust_widths))
            self._tw_width_mean = float(np.mean(cust_widths))
        
        self._demands_mean = float(np.mean(self.demands_kg))
        self._demands_std = float(np.std(self.demands_kg))

        # Spatial Density
        cust_dist = self.dist_matrix[1:, 1:]
        if cust_dist.shape[0] > 1:
            temp_dist = cust_dist.copy()
            np.fill_diagonal(temp_dist, np.inf)
            min_dists = np.min(temp_dist, axis=1)
            avg_min_dist = np.mean(min_dists)
            self._static_spatial_density = 1000.0 / (avg_min_dist + 1.0)
            
    @property
    def static_tw_tightness(self) -> float: return self._static_tw_tightness
    @property
    def static_spatial_density(self) -> float: return self._static_spatial_density
    @property
    def demands_mean(self) -> float: return self._demands_mean
    @property
    def demands_std(self) -> float: return self._demands_std
    @property
    def tw_width_mean(self) -> float: return self._tw_width_mean
    @property
    def num_nodes(self) -> int: return len(self.node_ids)

    def get_travel_time(self, from_node: int, to_node: int, v_type_id: int) -> float:
        return self.super_time_matrix[v_type_id, from_node, to_node]


# ==========================================
# PART 3: SOLUTION REPRESENTATION
# ==========================================
@dataclass
class Route:
    vehicle_type: VehicleType 
    node_sequence: List[int]  
    
    total_dist_meters: float = 0.0
    total_duration_min: float = 0.0
    total_wait_time_min: float = 0.0
    total_load_kg: float = 0.0
    total_load_cbm: float = 0.0

    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    
    is_time_feasible: bool = True
    is_capacity_feasible: bool = True
    is_duration_feasible: bool = True
    is_preference_feasible: bool = True
    
    def update_centroid(self, data: ProblemData):
        if not self.node_sequence:
            self.centroid_lat, self.centroid_lon = data.coords[0]
        else:
            coords = data.coords[self.node_sequence]
            m = np.mean(coords, axis=0)
            self.centroid_lat, self.centroid_lon = m[0], m[1]

    def clone(self) -> 'Route':
        return Route(
            vehicle_type=self.vehicle_type,
            node_sequence=self.node_sequence[:], # Fast slice copy
            total_dist_meters=self.total_dist_meters,
            total_duration_min=self.total_duration_min,
            total_wait_time_min=self.total_wait_time_min,
            total_load_kg=self.total_load_kg,
            total_load_cbm=self.total_load_cbm,
            centroid_lat=self.centroid_lat,
            centroid_lon=self.centroid_lon,
            is_time_feasible=self.is_time_feasible,
            is_capacity_feasible=self.is_capacity_feasible,
            is_duration_feasible=self.is_duration_feasible,
            is_preference_feasible=self.is_preference_feasible
        )

    @property
    def cost(self) -> float:
        v = self.vehicle_type
        dist_km = self.total_dist_meters / 1000.0
        time_hour = self.total_duration_min / 60.0
        return v.fixed_cost + (dist_km * v.cost_per_km) + (time_hour * v.cost_per_hour)

    @property
    def capacity_utilization(self) -> float:
        v = self.vehicle_type
        if v.capacity_kg <= 0 or v.capacity_cbm <= 0: return 0.0
        return max(self.total_load_kg / v.capacity_kg, self.total_load_cbm / v.capacity_cbm)

    @property
    def wait_time_ratio(self) -> float:
        if self.total_duration_min < 1e-6: return 0.0
        return self.total_wait_time_min / self.total_duration_min

@dataclass
class RvrpState:
    routes: List[Route]       
    unassigned: List[int]     

    def copy(self) -> 'RvrpState':
        return RvrpState([r.clone() for r in self.routes], self.unassigned[:])

    def objective(self) -> float:
        total_op_cost = 0.0
        util_penalty = 0.0
        for r in self.routes:
            v = r.vehicle_type
            dist_km = r.total_dist_meters / 1000.0
            time_hour = r.total_duration_min / 60.0
            r_cost = v.fixed_cost + (dist_km * v.cost_per_km) + (time_hour * v.cost_per_hour)
            total_op_cost += r_cost
            
            util = max(r.total_load_kg / v.capacity_kg, r.total_load_cbm / v.capacity_cbm)
            if util < 0.5:
                util_penalty += (0.5 - util) * v.fixed_cost * 5.0
                
        return total_op_cost + util_penalty

    @property
    def min_capacity_utilization(self) -> float:
        if not self.routes: return 0.0
        return min(r.capacity_utilization for r in self.routes)

    @property
    def max_capacity_utilization(self) -> float:
        if not self.routes: return 0.0
        return max(r.capacity_utilization for r in self.routes)

    @property
    def mean_capacity_utilization(self) -> float:
        if not self.routes: return 0.0
        return float(np.mean([r.capacity_utilization for r in self.routes]))
    
    @property
    def max_wait_time_ratio(self) -> float:
        if not self.routes: return 0.0
        return max(r.wait_time_ratio for r in self.routes)

# ==========================================
# PART 4: PPO AGENT STATE
# ==========================================
@dataclass
class PPOState:
    search_progress: float      
    stagnation_norm: float      
    best_cost_norm: float       
    current_cost_norm: float    
    improvement_history: float  
    
    demands_mean: float         
    demands_std: float          
    tw_width_mean: float        
    tw_tightness: float         
    spatial_density: float      
    
    min_cap_utilization: float  
    mean_cap_utilization: float 
    max_cap_utilization: float
    max_wait_time_ratio: float

    num_routes_norm: float      
    num_unassigned_norm: float      
    
    destroy_probs: np.ndarray   
    repair_probs: np.ndarray    
    
    def to_array(self) -> np.ndarray:
        scalars = np.array([
            self.search_progress, self.stagnation_norm, self.best_cost_norm,
            self.current_cost_norm, self.improvement_history, self.demands_mean,
            self.demands_std, self.tw_width_mean, self.tw_tightness,
            self.spatial_density, self.min_cap_utilization, self.mean_cap_utilization,
            self.max_cap_utilization, self.max_wait_time_ratio, self.num_routes_norm,
            self.num_unassigned_norm
        ], dtype=np.float32)
        return np.concatenate([scalars, self.destroy_probs.astype(np.float32), self.repair_probs.astype(np.float32)])

    @staticmethod
    def get_observation_size(n_destroy: int, n_repair: int) -> int:
        return 16 + n_destroy + n_repair