# file: ppo/rvrpenv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from core.data_structures import ProblemData, PPOState, RvrpState
from core.real_data_loader import RealDataLoader
from config import PPOConfig, ALNSConfig
import alns as vrp
from alns.alns4ppo import ALNS4PPO

ppo_cfg = PPOConfig()
alns_cfg = ALNSConfig()

class RVRPEnvironment(gym.Env):
    def __init__(self, order_csv_path, truck_csv_path, is_test_mode=False, override_depot_id=None):
        super().__init__()
        self.loader = RealDataLoader()
        self.problem_data: ProblemData = self.loader.load_day_data(order_csv_path, truck_csv_path, override_depot_id)
        self.is_test_mode = is_test_mode
        self.alns = ALNS4PPO()
        
        self.alns.reset_opt()
        self._register_operators()
        self.d_op_num = len(self.alns.destroy_operators)
        self.r_op_num = len(self.alns.repair_operators)
        
        # [OPTIMIZATION] Pre-calculate and cache indices of route-based operators
        self.route_op_indices = np.array([
            i for i, (name, _) in enumerate(self.alns.destroy_operators)
            if any(k in name for k in ["route", "sequence", "low_util", "eliminate"])
        ])

        # [OPTIMIZATION] Cache static observation features
        pd = self.problem_data
        self.static_obs_part = np.array([
            pd.demands_mean, pd.demands_std, pd.tw_width_mean,
            pd.static_tw_tightness, pd.static_spatial_density
        ], dtype=np.float32)

        self.stop_counter = 0
        self.last_improvement = 0.0

        self.destroy_usage = np.zeros(self.d_op_num, dtype=np.float32)
        self.repair_usage = np.zeros(self.r_op_num, dtype=np.float32)

        self.current_solution: RvrpState = None
        self.best_solution: RvrpState = None
        self.ppo_state: PPOState = None

        self.action_space = spaces.MultiDiscrete([self.d_op_num, self.r_op_num, 2, 2])
        obs_size = PPOState.get_observation_size(self.d_op_num, self.r_op_num)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.iters = 0
        self.stagnation_counter = 0
        
        # [OPTIMIZATION] Use fast init for training, slow high-quality init for testing
        if self.is_test_mode:
            self.init_solution = vrp.initial.clarke_wright_heterogeneous(self.problem_data)
        else:
            # Assumes you add a simple one-for-one or greedy init function
            self.init_solution = vrp.initial.one_for_one(self.problem_data)
        
        self.current_solution = self.init_solution.copy()
        self.pre_solution = self.init_solution.copy()
        # Initialize centroids once
        for r in self.current_solution.routes: r.update_centroid(self.problem_data)
        
        self.best_solution = self.current_solution.copy()
        self.initial_cost = max(1.0, self.init_solution.objective())
        
        return self._extract_features().to_array(), {}

    def step(self, action):
        d_idx, r_idx, accept, stop = action
        self.iters += 1
        
        self.destroy_usage[d_idx] += 1
        self.repair_usage[r_idx] += 1
        
        start_time_alns = time.time() 
        
        # 1. Execute ALNS Iteration
        self.pre_solution, self.current_solution = self.alns.iterate(
            self.current_solution, self.pre_solution, d_idx, r_idx, accept, 
            data=self.problem_data
        )
        alns_execution_time = time.time() - start_time_alns # Thời gian chạy ALNS

        # 2. Calculate Costs (uses optimized objective call)
        prev_cost = self.pre_solution.objective()
        curr_cost = self.current_solution.objective()
        best_cost = self.best_solution.objective()
        
        # Improvement: Pos = Better, Neg = Worse
        step_improvement = (prev_cost - curr_cost) / max(1e-6, prev_cost)
        self.last_improvement = step_improvement
        
        # 3. Global Best Tracking
        if curr_cost < best_cost:
            self.best_solution = self.current_solution.copy()
            self.stagnation_counter = 0
            reward_best = 4.0 * (best_cost - curr_cost) / best_cost
        else:
            self.stagnation_counter += 1
            reward_best = 0.0

        # 4. Reward & Features
        reward = self._calculate_reward(step_improvement, accept, self.stagnation_counter, alns_execution_time) + reward_best
        ppo_obs = self._extract_features()
        
        terminated = self.stagnation_counter >= ppo_cfg.stop_threshold or self.iters >= alns_cfg.num_iterations or stop == 1
        
        # [DEBUG] Only print if the step is unusually slow to avoid console lag
        # if alns_execution_time > 0.1:
        #     print(f"[ENV_SLOW_STEP] {self.iters}: {alns_execution_time:.4f}s, Action {d_idx, r_idx, accept, stop}")

        return ppo_obs.to_array(), reward, terminated, False, {}

    def _extract_features(self) -> PPOState:
        sol = self.current_solution
        # [OPTIMIZATION] Combine cached static features with dynamic ones
        return PPOState(
            search_progress = self.iters / ALNSConfig().num_iterations,
            stagnation_norm = min(1.0, self.stagnation_counter / 50.0),
            best_cost_norm = self.best_solution.objective() / self.initial_cost,
            current_cost_norm = sol.objective() / self.initial_cost,
            improvement_history = self.last_improvement,
            demands_mean = self.static_obs_part[0],
            demands_std = self.static_obs_part[1],
            tw_width_mean = self.static_obs_part[2],
            tw_tightness = self.static_obs_part[3],
            spatial_density = self.static_obs_part[4],
            max_wait_time_ratio = sol.max_wait_time_ratio,
            min_cap_utilization = sol.min_capacity_utilization,
            mean_cap_utilization = sol.mean_capacity_utilization,
            max_cap_utilization = sol.max_capacity_utilization,
            num_routes_norm = len(sol.routes) / self.problem_data.num_nodes,
            num_unassigned_norm = len(sol.unassigned) / self.problem_data.num_nodes,
            destroy_probs = np.zeros(self.d_op_num), # Pre-allocate or track
            repair_probs = np.zeros(self.r_op_num)
        )

    # def _calculate_reward(self, improvement, accept_action, stagnation):
    #     # 1. Cost Improvement
    #     reward_cost = improvement * ppo_cfg.reward_cost_scale
    #     # 2. Utilization Reward
    #     prev_util = self.pre_solution.mean_capacity_utilization
    #     curr_util = self.current_solution.mean_capacity_utilization
    #     reward_util = (curr_util - prev_util) * ppo_cfg.reward_util_lambda
    #     # 3. Exploration Bonus
    #     reward_explore = 0.0
    #     if improvement <= 0 and accept_action == 1 and stagnation > 10:
    #         reward_explore = 0.05 * (stagnation / 50.0)
    #     return reward_cost + reward_util + reward_explore

    def _calculate_reward(self, improvement, accept_action, stagnation, alns_time):
        """
        Hybrid Reward: Cost + Utilization + Time Penalty + Consistency Penalty + Exploration Bonus
        """
        reward_cost = improvement * ppo_cfg.reward_cost_scale
        
        prev_util = self.pre_solution.mean_capacity_utilization
        curr_util = self.current_solution.mean_capacity_utilization
        reward_util = (curr_util - prev_util) * ppo_cfg.reward_util_lambda
        
        # --- TIME PENALTIES ---
        time_penalty = ppo_cfg.base_step_penalty # Base penalty per RL step
        # Penalty on ALNS execution time
        if alns_time > ppo_cfg.operator_time_limit:
            time_penalty += (alns_time - ppo_cfg.operator_time_limit) * ppo_cfg.operator_penalty_scale
        
        # Phạt nặng hơn nếu về cuối game mà vẫn ko cải thiện
        search_progress = self.iters / ALNSConfig().num_iterations
        if search_progress > 0.8 and improvement <= 0:
            time_penalty += 0.05 
        
        # Exploration Bonus
        reward_explore = 0.0
        if improvement < 0 and accept_action == 1 and stagnation > 15:
            reward_explore = 0.05 * (stagnation / 50.0)
        
        return (reward_cost + reward_util + reward_explore 
                - time_penalty)

    def _register_operators(self):
        r_small = 0.1
        r_medium = 0.3
        r_large = 0.5
        
        # --- 1. RANDOM REMOVAL (Basic Diversification) ---
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=r_small), name="random_small")
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=r_medium), name="random_medium")
        self.alns.add_destroy_operator(vrp.create_random_customer_removal_operator(self.problem_data, ratio=r_large), name="random_large")
        # --- 2. WORST REMOVAL (Cost Reduction Focus) ---
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=r_small), name="worst_small")
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=r_medium), name="worst_medium")
        self.alns.add_destroy_operator(vrp.create_worst_removal_operator(self.problem_data, ratio=r_large), name="worst_large")
        # --- 3. STRING REMOVAL (Spatial/Sequence Focus) ---
        self.alns.add_destroy_operator(vrp.create_string_removal_operator(self.problem_data, ratio=r_small), name="string_small")
        self.alns.add_destroy_operator(vrp.create_string_removal_operator(self.problem_data, ratio=r_medium), name="string_medium")
        self.alns.add_destroy_operator(vrp.create_string_removal_operator(self.problem_data, ratio=r_large), name="string_large")
        # --- 4. RELATED REMOVAL (Shaw - Similarity Focus) ---
        self.alns.add_destroy_operator(vrp.create_related_removal_operator(self.problem_data, ratio=r_small), name="related_small")
        self.alns.add_destroy_operator(vrp.create_related_removal_operator(self.problem_data, ratio=r_medium), name="related_medium")
        self.alns.add_destroy_operator(vrp.create_related_removal_operator(self.problem_data, ratio=r_large), name="related_large")
        # --- 5. ROUTE REMOVAL (Structure Focus - High Impact) ---
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=r_small), name="route_small")
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=r_medium), name="route_medium")
        self.alns.add_destroy_operator(vrp.create_random_route_removal_operator(self.problem_data, ratio=r_large), name="route_large")
        # --- 6. SEQUENCE REMOVAL (Sub-tour Focus) ---
        self.alns.add_destroy_operator(vrp.create_sequence_removal_operator(self.problem_data, ratio=r_small), name="sequence_small")
        self.alns.add_destroy_operator(vrp.create_sequence_removal_operator(self.problem_data, ratio=r_medium), name="sequence_medium")
        self.alns.add_destroy_operator(vrp.create_sequence_removal_operator(self.problem_data, ratio=r_large), name="sequence_large")
        # --- 7. LOW UTILIZATION ROUTE REMOVAL (Optimization Focus) ---
        self.alns.add_destroy_operator(vrp.create_low_utilization_route_removal_operator(self.problem_data, ratio=r_small), name="low_util_small")
        self.alns.add_destroy_operator(vrp.create_low_utilization_route_removal_operator(self.problem_data, ratio=r_medium), name="low_util_medium")
        self.alns.add_destroy_operator(vrp.create_low_utilization_route_removal_operator(self.problem_data, ratio=r_large), name="low_util_large")
        # --- 8. ELIMINATE SMALL ROUTES (Cleaner) ---
        self.alns.add_destroy_operator(vrp.create_eliminate_small_route_operator(self.problem_data, min_stops=3), name="eliminate_small")
        
        # --- REPAIR OPERATORS ---
        # 1. Greedy (Baseline)
        self.alns.add_repair_operator(vrp.create_greedy_repair_operator(self.problem_data), name="greedy_repair")
        # 2. Criticality (Composite Score)
        self.alns.add_repair_operator(vrp.create_criticality_repair_operator(self.problem_data), name="criticality_repair")
        # 3. Regret-2 (Lookahead 2)
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data, k=2), name="regret_2_repair")
        # 4. Regret-3 (Lookahead 3)
        self.alns.add_repair_operator(vrp.create_regret_repair_operator(self.problem_data, k=3), name="regret_3_repair")
        # 5. GRASP (Randomized)
        self.alns.add_repair_operator(vrp.create_grasp_repair_operator(self.problem_data, rcl_size=3), name="grasp_3_repair")
        # 6. Farthest (Spatial - Outside In)
        self.alns.add_repair_operator(vrp.create_farthest_insertion_repair_operator(self.problem_data), name="farthest_repair")
        # 7. [NEW] Largest Demand (Bin Packing Focus)
        self.alns.add_repair_operator(vrp.create_largest_demand_repair_operator(self.problem_data), name="largest_demand_repair")
        # 8. [NEW] Earliest TW (Scheduling Focus)
        self.alns.add_repair_operator(vrp.create_earliest_tw_repair_operator(self.problem_data), name="earliest_tw_repair")
        # 9. [NEW] Closest (Spatial - Depot Focus)
        self.alns.add_repair_operator(vrp.create_closest_to_depot_repair_operator(self.problem_data), name="closest_depot_repair")

    def valid_action_mask(self) -> np.ndarray:
        # [OPTIMIZATION] Vectorized masking
        mask = np.ones(self.d_op_num + self.r_op_num + 2 + 2, dtype=bool)
        if len(self.current_solution.routes) == 0:
            mask[self.route_op_indices] = False
        return mask