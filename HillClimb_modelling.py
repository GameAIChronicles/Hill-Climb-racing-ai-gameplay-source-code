"""
This file or module train the model and save the model in data/models/Dino_model folder.
"""

# Import Library
from ENV_Hill_climb import ENV_hill_climb
import os
from stable_baselines3 import PPO


log_path = os.path.join('data', 'logs')  # Path of the log folder

env = ENV_hill_climb()  # Costume environment


model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log=log_path)  # Setting up the algorithm
env.reset()
Time = 1


model.learn(total_timesteps=Time, reset_num_timesteps=False, tb_log_name='PPO')  # Training the model
model.save(os.path.join('data', 'models', 'HillClimbModel3', f'model_PPO_{Time}_new.zip'))  # Saving the model


"""
After the model is trained the model will be saved to the folder. It could be used for prediction later.
"""