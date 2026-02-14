import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

@dataclass
class PathConfig:
    now_str = datetime.now().strftime("%m%d_%H%M")
    ORDER_PATH: str = "inputs/CleanData/Split_TransportOrder_2524.csv"
    TRUCK_PATH: str = "inputs/MasterData/TruckMaster.csv"
    DISTANCE_TIME_PATH: str = "inputs/DistTimeMatrix"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "models"
    MODEL_PATH: str = "models/VRPTW_1225_1231/"

    TEMP_DATA_DIR: str = "inputs/temp_days"
    FINAL_REPORT_DIR: str = f"results/{now_str}"
    INFERENCE_MODEL_PATH: str = "models/model_170_125K/vrp_model_125000_steps.zip"

@dataclass
class PPOConfig:
    train_seed: int = 2025
    num_envs: int = 1               
    
    n_steps: int = 2048           
    batch_size: int = 256
    n_epochs: int = 10            
    learning_rate: float = 3e-4   
    ent_coef: float = 0.01        
    
    current_trained_timesteps: int = 0
    total_timesteps: int = 1_500_000 
    save_freq: int = 5000     
    
    reward_cost_scale: float = 5.0  
    reward_util_lambda: float = 0.5 
    stop_threshold: int = 50       

    base_step_penalty: float = 0.005   
    operator_time_limit: float = 2.0
    operator_penalty_scale: float = 0.01
    
    device: str = "auto"

    def __post_init__(self):
        now_str = datetime.now().strftime("%m%d_%H%M")
        self.run_name = f"VRPTW_{now_str}"
        self.tensorboard_log = os.path.join(PathConfig.LOGS_DIR, "tb")
        self.model_save_path = os.path.join(PathConfig.MODELS_DIR, self.run_name)
        self.monitor_path = os.path.join(self.model_save_path, "monitor.csv")

@dataclass
class ALNSConfig:
    num_iterations: int = 100