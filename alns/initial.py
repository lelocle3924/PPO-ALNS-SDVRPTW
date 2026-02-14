
import numpy as np
from numba import njit
from typing import List, Tuple, Dict, Optional, Any
from core.data_structures import RvrpState, Route, ProblemData, VehicleType

# ============================================================
# 0. NUMBA ACCELERATED KERNELS
# ============================================================

@njit(fastmath=True, cache=True)
def _evaluate_sequence_numba(dist_mat, time_mat, tw, service, dem_kg, dem_cbm, 
                             cap_kg, cap_cbm, max_dur, depot_open, depot_close, sequence):
    total_kg, total_cbm = 0.0, 0.0
    for i in range(sequence.shape[0]):
        node = sequence[i]
        total_kg += dem_kg[node]
        total_cbm += dem_cbm[node]
    if total_kg > cap_kg or total_cbm > cap_cbm: return -1, 0.0, 0.0, 0.0, total_kg, total_cbm

    curr_time, total_dist, total_wait, prev = depot_open, 0.0, 0.0, 0
    for i in range(sequence.shape[0]):
        node = sequence[i]
        total_dist += dist_mat[prev, node]
        curr_time += time_mat[prev, node]
        if curr_time > tw[node, 1]: return -2, 0.0, 0.0, 0.0, total_kg, total_cbm
        if curr_time < tw[node, 0]:
            total_wait += (tw[node, 0] - curr_time)
            curr_time = tw[node, 0]
        curr_time += service[node]
        prev = node
    
    total_dist += dist_mat[prev, 0]
    curr_time += time_mat[prev, 0]
    if curr_time > depot_close: return -3, 0.0, 0.0, 0.0, total_kg, total_cbm
    dur = curr_time - depot_open
    if dur > max_dur: return -4, 0.0, 0.0, 0.0, total_kg, total_cbm
    return 1, total_dist, dur, total_wait, total_kg, total_cbm

def check_sequence_feasibility(data: ProblemData, vehicle: VehicleType, node_sequence: List[int]):
    if not node_sequence: return True, None, "Empty"
    seq_arr = np.array(node_sequence, dtype=np.int32)
    status, dist, dur, wait, kg, cbm = _evaluate_sequence_numba(
        data.dist_matrix, data.super_time_matrix[vehicle.type_id], data.time_windows, 
        data.service_times, data.demands_kg, data.demands_cbm, vehicle.capacity_kg, 
        vehicle.capacity_cbm, data.max_route_duration, data.time_windows[0][0], 
        data.time_windows[0][1], seq_arr
    )
    if status == 1:
        return True, {"total_dist_meters": dist, "total_duration_min": dur, "total_wait_time_min": wait, "total_load_kg": kg, "total_load_cbm": cbm}, "OK"
    return False, None, str(status)


def _find_best_vehicle_for_sequence(data: ProblemData, sequence: list[int]) -> tuple[VehicleType, dict]:
    """
    [OPTIMIZED] Uses Numba engine to find best vehicle.
    """
    # Sort fleet by fixed cost to find the cheapest feasible option
    sorted_fleet = sorted(data.vehicle_types, key=lambda v: v.fixed_cost)
    
    for v in sorted_fleet:
        # Accelerated call to Numba simulation
        is_feasible, metrics, _ = check_sequence_feasibility(data, v, sequence)
        if is_feasible:
            return v, metrics
    return None, None

def one_for_one(data: ProblemData) -> RvrpState:
    """
    Dummy initialization for 0.001s resets during training.
    """
    routes = []
    # Use the cheapest vehicle available
    v_type = min(data.vehicle_types, key=lambda x: x.fixed_cost)
    
    for i in range(1, data.num_nodes):
        # Create single-node route
        r = Route(v_type, [i])
        is_f, metrics, _ = check_sequence_feasibility(data, v_type, [i])
        if metrics:
            r.total_dist_meters = metrics['total_dist_meters']
            r.total_duration_min = metrics['total_duration_min']
            r.total_load_kg = metrics['total_load_kg']
            r.total_load_cbm = metrics['total_load_cbm']
        r.update_centroid(data)
        routes.append(r)
        
    return RvrpState(routes, [])

def clarke_wright_heterogeneous(data: ProblemData) -> RvrpState:

    customers = [i for i in range(1, data.num_nodes)]
    initial_routes = []
    node_to_route = {}
    
    # 1. Initial State: Best Fit for single nodes
    for cust in customers:
        v, metrics = _find_best_vehicle_for_sequence(data, [cust])
        if v:
            r = Route(v, [cust], 
                      total_dist_meters=metrics['total_dist_meters'],
                      total_duration_min=metrics['total_duration_min'],
                      total_wait_time_min=metrics['total_wait_time_min'],
                      total_load_kg=metrics['total_load_kg'],
                      total_load_cbm=metrics['total_load_cbm'])
            r.update_centroid(data)
            initial_routes.append(r)
            node_to_route[cust] = r

    # 2. Savings Calculation
    savings = []
    dm = data.dist_matrix
    for i in customers:
        if i not in node_to_route: continue
        for j in range(i + 1, data.num_nodes):
            if j not in node_to_route: continue
            s = dm[i, 0] + dm[0, j] - dm[i, j]
            if s > 0:
                savings.append((s, i, j))
    
    savings.sort(key=lambda x: x[0], reverse=True)

    # 3. Merge Loop
    for _, i, j in savings:
        r_i = node_to_route.get(i)
        r_j = node_to_route.get(j)
        
        if r_i is None or r_j is None or r_i is r_j:
            continue
            
        # Check endpoints
        i_end = (r_i.node_sequence[-1] == i)
        j_start = (r_j.node_sequence[0] == j)
        
        if i_end and j_start:
            merge_seq = r_i.node_sequence + r_j.node_sequence
            new_v, metrics = _find_best_vehicle_for_sequence(data, merge_seq)
            
            if new_v:
                r_i.node_sequence = merge_seq
                r_i.vehicle_type = new_v
                r_i.total_dist_meters = metrics['total_dist_meters']
                r_i.total_duration_min = metrics['total_duration_min']
                r_i.total_load_kg = metrics['total_load_kg']
                r_i.total_load_cbm = metrics['total_load_cbm']
                r_i.update_centroid(data)
                
                for node in r_j.node_sequence:
                    node_to_route[node] = r_i
                r_j.node_sequence = []
                
    final_routes = [r for r in initial_routes if len(r.node_sequence) > 0]
    unassigned = [i for i in range(1, data.num_nodes) if i not in node_to_route]
    
    return RvrpState(final_routes, unassigned)

def neighbors(dist_matrix, customer_idx):
    locations = np.argsort(dist_matrix[customer_idx])
    return locations[locations != 0]