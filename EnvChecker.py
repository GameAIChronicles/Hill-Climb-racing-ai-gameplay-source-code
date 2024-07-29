from stable_baselines3.common.env_checker import check_env
from ENV_Hill_climb import ENV_hill_climb

env = ENV_hill_climb()
check_env(env=env)