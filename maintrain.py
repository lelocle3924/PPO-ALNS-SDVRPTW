import argparse
import os
import multiprocessing
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from core import LightLogger
from config import PPOConfig, PathConfig
from ppo.rvrpenv import RVRPEnvironment

ppo_cfg = PPOConfig()
path_cfg = PathConfig()
logger = LightLogger(name="Main")

def mask_fn(env):
    return env.valid_action_mask()

def make_env(is_test_mode=False):
    def _init():
        env = RVRPEnvironment(
            order_csv_path=path_cfg.ORDER_PATH,
            truck_csv_path=path_cfg.TRUCK_PATH,
            is_test_mode=is_test_mode
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def train():
    os.makedirs(ppo_cfg.model_save_path, exist_ok=True)
    os.makedirs(ppo_cfg.tensorboard_log, exist_ok=True)
    
    # Check for CUDA
    device = ppo_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize Parallel Envs
    logger.info(f"Spawning {ppo_cfg.num_envs} workers...")
    if ppo_cfg.num_envs > 1:
        train_env = SubprocVecEnv([make_env(is_test_mode=False) for _ in range(ppo_cfg.num_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([make_env(is_test_mode=False) for _ in range(ppo_cfg.num_envs)])
    train_env = VecMonitor(train_env, ppo_cfg.monitor_path)

    steps_to_train = ppo_cfg.total_timesteps
    reset_timesteps = True

    # Load Model or Create New
    if path_cfg.MODEL_PATH and os.path.exists(path_cfg.MODEL_PATH):
        logger.info(f"Resuming training from {path_cfg.MODEL_PATH}...")
        logger.info(f"   Current Progress: {ppo_cfg.current_trained_timesteps} steps")
        model = MaskablePPO.load(path_cfg.MODEL_PATH, 
        env=train_env, 
        device=device, 
        tensorboard_log=ppo_cfg.tensorboard_log,
        learning_rate=ppo_cfg.learning_rate
        )
        steps_to_train = ppo_cfg.total_timesteps - ppo_cfg.current_trained_timesteps
        if steps_to_train <= 0:
            logger.warning("Target timesteps reached already. Training might stop immediately.")
            steps_to_train = 10000
            
        reset_timesteps = False
    else:
        logger.info("Initializing NEW MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy", 
            train_env, 
            n_steps=ppo_cfg.n_steps, 
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            learning_rate=ppo_cfg.learning_rate,
            verbose=1, 
            device=device,
            tensorboard_log=ppo_cfg.tensorboard_log,
            ent_coef=ppo_cfg.ent_coef,
        )
        steps_to_train = ppo_cfg.total_timesteps
        reset_timesteps = True


    checkpoint_callback = CheckpointCallback(
        save_freq=max(ppo_cfg.save_freq // ppo_cfg.num_envs, 1), 
        save_path=ppo_cfg.model_save_path, 
        name_prefix="vrp_model"
    )

    try:
        model.learn(
            total_timesteps=steps_to_train,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
        
        final_path = os.path.join(ppo_cfg.model_save_path, f"final_vrp_model_{ppo_cfg.total_timesteps}_steps.zip")
        model.save(final_path)
        logger.info(f"Training Complete. Saved to: {final_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted manually. Saving current state...")
        model.save(os.path.join(ppo_cfg.model_save_path, "interrupted_model.zip"))
    finally:
        train_env.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()