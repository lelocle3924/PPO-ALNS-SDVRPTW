import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from numba import njit
from core.data_structures import RvrpState, Route, ProblemData, VehicleType

# ============================================================
# 0. NUMBA ACCELERATED KERNELS
# ============================================================


@njit(fastmath=True, cache=True)
def _evaluate_insertion_numba(
    dist_mat, time_mat, tw, service, dem_kg, dem_cbm,
    cap_kg, cap_cbm, max_dur, depot_open, depot_close,
    sequence, customer_idx, insert_pos
):
    """
    Evaluates a single insertion at a specific position.
    Returns: (Status, dist, dur, wait, kg, cbm)
    """
    total_kg = 0.0
    total_cbm = 0.0
    
    # 1. Capacity
    for i in range(sequence.shape[0]):
        n = sequence[i]
        total_kg += dem_kg[n]
        total_cbm += dem_cbm[n]
    total_kg += dem_kg[customer_idx]
    total_cbm += dem_cbm[customer_idx]
    
    if total_kg > cap_kg or total_cbm > cap_cbm:
        return -1, 0.0, 0.0, 0.0, total_kg, total_cbm

    # 2. Time/Dist Simulation
    curr_time = depot_open
    total_dist = 0.0
    total_wait = 0.0
    prev = 0
    
    seq_len = sequence.shape[0]
    for i in range(seq_len + 1):
        if i == insert_pos:
            node = customer_idx
        else:
            idx = i if i < insert_pos else i - 1
            node = sequence[idx]
            
        total_dist += dist_mat[prev, node]
        curr_time += time_mat[prev, node]
        
        if curr_time > tw[node, 1]: return -2, 0.0, 0.0, 0.0, total_kg, total_cbm
        if curr_time < tw[node, 0]:
            total_wait += (tw[node, 0] - curr_time)
            curr_time = tw[node, 0]
            
        curr_time += service[node]
        prev = node
        
    # Return to Depot
    total_dist += dist_mat[prev, 0]
    curr_time += time_mat[prev, 0]
    
    if curr_time > depot_close: return -3, 0.0, 0.0, 0.0, total_kg, total_cbm
    dur = curr_time - depot_open
    if dur > max_dur: return -4, 0.0, 0.0, 0.0, total_kg, total_cbm
    
    return 1, total_dist, dur, total_wait, total_kg, total_cbm

# ============================================================
# 1. CORE SEARCH ENGINE
# ============================================================

def find_best_insertion_for_route(data: ProblemData, route: Route, customer: int) -> Tuple[float, int, Dict, VehicleType]:
    """
    [OPTIMIZED] Finds the cheapest position for a customer in a route.
    Handles heterogeneous fleet upgrades if current vehicle fails.
    """
    best_cost = float('inf')
    best_pos = -1
    best_metrics = None
    best_veh = None
    
    # 1. Check Preferences first (Python side)
    allowed = data.allowed_vehicles[customer]
    
    # 2. Try vehicles starting from current one, then upgrading
    # We sort fleet by capacity to find the smallest feasible upgrade
    potential_vehicles = [route.vehicle_type]
    for v in data.vehicle_types:
        if v.capacity_kg > route.vehicle_type.capacity_kg:
            potential_vehicles.append(v)
    
    potential_vehicles.sort(key=lambda x: x.fixed_cost)

    seq_arr = np.array(route.node_sequence, dtype=np.int32)
    
    for v in potential_vehicles:
        if v.type_id not in allowed: continue
        
        time_slice = data.super_time_matrix[v.type_id]
        v_best_pos = -1
        v_best_cost = float('inf')
        v_best_metrics = None
        
        # Inner Loop: Try every position
        for pos in range(len(route.node_sequence) + 1):
            status, dist, dur, wait, kg, cbm = _evaluate_insertion_numba(
                data.dist_matrix, time_slice, data.time_windows, data.service_times,
                data.demands_kg, data.demands_cbm, v.capacity_kg, v.capacity_cbm,
                data.max_route_duration, data.time_windows[0][0], data.time_windows[0][1],
                seq_arr, customer, pos
            )
            
            if status == 1:
                # Calculate Cost
                cost = v.fixed_cost + (dist/1000.0 * v.cost_per_km) + (dur/60.0 * v.cost_per_hour)
                # We want cheapest insertion for THIS vehicle
                if cost < v_best_cost:
                    v_best_cost = cost
                    v_best_pos = pos
                    v_best_metrics = {
                        "total_dist_meters": dist, "total_duration_min": dur,
                        "total_wait_time_min": wait, "total_load_kg": kg, "total_load_cbm": cbm
                    }
        
        # If this vehicle found ANY feasible position, we take it and stop upgrading
        # (Since fleet is sorted by cost/capacity)
        if v_best_pos != -1:
            return v_best_cost, v_best_pos, v_best_metrics, v
            
    return float('inf'), -1, None, None

def _get_candidate_routes(data: ProblemData, routes: List[Route], customer: int, limit: int = 40):
    if len(routes) <= limit:
        return range(len(routes))
    
    c = data.coords[customer]
    dists = []
    for i, r in enumerate(routes):
        dists.append(((r.centroid_lat - c[0])**2 + (r.centroid_lon - c[1])**2, i))
    
    dists.sort(key=lambda x: x[0])
    return [x[1] for x in dists[:limit]]

def _try_create_new_route(data: ProblemData, customer: int) -> Optional[Route]:
    sorted_fleet = sorted(data.vehicle_types, key=lambda x: x.fixed_cost)
    allowed = data.allowed_vehicles[customer]
    
    seq_arr = np.array([customer], dtype=np.int32)
    empty_arr = np.array([], dtype=np.int32)
    
    for v in sorted_fleet:
        if v.type_id not in allowed: continue
        status, dist, dur, wait, kg, cbm = _evaluate_insertion_numba(
            data.dist_matrix, data.super_time_matrix[v.type_id], data.time_windows, data.service_times,
            data.demands_kg, data.demands_cbm, v.capacity_kg, v.capacity_cbm,
            data.max_route_duration, data.time_windows[0][0], data.time_windows[0][1],
            empty_arr, customer, 0
        )
        if status == 1:
            r = Route(v, [customer], dist, dur, wait, kg, cbm)
            r.update_centroid(data)
            return r
    return None

# ============================================================
# 3. REPAIR OPERATORS
# ============================================================

def create_greedy_repair_operator(data: ProblemData):
    def greedy_repair(state: RvrpState, rng, data: ProblemData):
        if rng is not None:
            rng.shuffle(state.unassigned)

        while state.unassigned:
            customer = state.unassigned.pop()
            best_increase = float('inf')
            best_move = None
            
            candidate_indices = _get_candidate_routes(data, state.routes, customer)
            
            for r_idx in candidate_indices:
                route = state.routes[r_idx]
                cost, pos, metrics, veh = find_best_insertion_for_route(data, route, customer)
                if pos != -1:
                    increase = cost - route.cost
                    if increase < best_increase:
                        best_increase = increase
                        best_move = (r_idx, pos, metrics, veh)
            
            new_r = _try_create_new_route(data, customer)
            if new_r and new_r.cost < best_increase:
                state.routes.append(new_r)
            elif best_move:
                r_idx, pos, m, v = best_move
                target = state.routes[r_idx]
                target.node_sequence.insert(pos, customer)
                target.vehicle_type, target.total_dist_meters = v, m["total_dist_meters"]
                target.total_duration_min, target.total_wait_time_min = m["total_duration_min"], m["total_wait_time_min"]
                target.total_load_kg, target.total_load_cbm = m["total_load_kg"], m["total_load_cbm"]
                target.update_centroid(data)
        return state
    return greedy_repair

def create_criticality_repair_operator(data: ProblemData):
    def criticality_repair(state: RvrpState, rng, data: ProblemData):
        state.unassigned.sort(key=lambda c: (data.demands_kg[c]/1000 + data.dist_matrix[0,c]/5000), reverse=True)
        return create_greedy_repair_operator(data)(state, rng, data)
    return criticality_repair

def create_regret_repair_operator(data: ProblemData, k: int = 2):
    def regret_repair(state: RvrpState, rng, data: ProblemData):
        while state.unassigned:
            best_regret = -1.0
            best_cust_move = None # (cust, r_idx, pos, metrics, vehicle)

            for cust in state.unassigned:
                insertions = []
                candidate_indices = _get_candidate_routes(data, state.routes, cust, limit=30)
                
                for r_idx in candidate_indices:
                    route = state.routes[r_idx]
                    cost, pos, m, v = find_best_insertion_for_route(data, route, cust)
                    if pos != -1: insertions.append((cost - route.cost, r_idx, pos, m, v))
                
                new_r = _try_create_new_route(data, cust)
                if new_r: insertions.append((new_r.cost, -1, 0, None, new_r))
                
                if not insertions: continue
                insertions.sort(key=lambda x: x[0])
                
                # Regret = Cost_K - Cost_1
                current_regret = insertions[min(k-1, len(insertions)-1)][0] - insertions[0][0]
                if current_regret > best_regret:
                    best_regret = current_regret
                    best_cust_move = (cust, *insertions[0][1:])
            
            if not best_cust_move: break
            cust, r_idx, pos, m, v = best_cust_move
            state.unassigned.remove(cust)
            if r_idx == -1: state.routes.append(v) # v is the new_route object
            else:
                target = state.routes[r_idx]
                target.node_sequence.insert(pos, cust)
                target.vehicle_type, target.total_dist_meters = v, m["total_dist_meters"]
                target.total_duration_min, target.total_wait_time_min = m["total_duration_min"], m["total_wait_time_min"]
                target.total_load_kg, target.total_load_cbm = m["total_load_kg"], m["total_load_cbm"]
                target.update_centroid(data)
        return state
    return regret_repair

def create_grasp_repair_operator(data: ProblemData, rcl_size: int = 3):
    def grasp_repair(state: RvrpState, rng, data: ProblemData):
        while state.unassigned:
            moves = []
            for cust in state.unassigned:
                candidate_indices = _get_candidate_routes(data, state.routes, cust, limit=20)
                for r_idx in candidate_indices:
                    route = state.routes[r_idx]
                    cost, pos, m, v = find_best_insertion_for_route(data, route, cust)
                    if pos != -1: moves.append((cost - route.cost, cust, r_idx, pos, m, v))
                new_r = _try_create_new_route(data, cust)
                if new_r: moves.append((new_r.cost, cust, -1, 0, None, new_r))
            
            if not moves: break
            moves.sort(key=lambda x: x[0])
            winner = moves[rng.integers(0, min(rcl_size, len(moves)))]
            
            cost, cust, r_idx, pos, m, v = winner
            state.unassigned.remove(cust)
            if r_idx == -1: state.routes.append(v)
            else:
                target = state.routes[r_idx]
                target.node_sequence.insert(pos, cust)
                target.vehicle_type, target.total_dist_meters = v, m["total_dist_meters"]
                target.total_duration_min, target.total_wait_time_min = m["total_duration_min"], m["total_wait_time_min"]
                target.total_load_kg, target.total_load_cbm = m["total_load_kg"], m["total_load_cbm"]
                target.update_centroid(data)
        return state
    return grasp_repair

# Helper for sorted operators
def _apply_sorted_repair(state, data, sorted_unassigned):
    state.unassigned = sorted_unassigned
    return create_greedy_repair_operator(data)(state, None, data)

def create_largest_demand_repair_operator(data: ProblemData):
    return lambda s, rng, data: _apply_sorted_repair(s, data, sorted(s.unassigned, key=lambda c: data.demands_kg[c], reverse=True))

def create_earliest_tw_repair_operator(data: ProblemData):
    return lambda s, rng, data: _apply_sorted_repair(s, data, sorted(s.unassigned, key=lambda c: data.time_windows[c][0]))

def create_closest_to_depot_repair_operator(data: ProblemData):
    return lambda s, rng, data: _apply_sorted_repair(s, data, sorted(s.unassigned, key=lambda c: data.dist_matrix[0,c]))

def create_farthest_insertion_repair_operator(data: ProblemData):
    return lambda s, rng, data: _apply_sorted_repair(s, data, sorted(s.unassigned, key=lambda c: data.dist_matrix[0,c], reverse=True))