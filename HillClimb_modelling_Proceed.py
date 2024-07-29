"""
This file or module train the model and save the model in data/models/Dino_model folder.
"""

# Import Library
from stable_baselines3 import PPO
from ENV_Hill_climb import ENV_hill_climb
import os
env = ENV_hill_climb()

model = PPO.load(os.path.join('data', 'models', 'HillClimbModel3', 'model_PPO_1_new.zip'), env=env)

model.learn(total_timesteps=1, reset_num_timesteps=False, tb_log_name='PPO')  # Training the model
model.save(os.path.join('data', 'models', 'HillClimbModel3', 'model_PPO_2_new.zip'))


"""
After the model is trained the model will be saved to the folder. It could be used for prediction later.
"""