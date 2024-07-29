from stable_baselines3 import PPO
from ENV_Hill_climb import ENV_hill_climb
import os
env = ENV_hill_climb()

model = PPO.load(os.path.join('data', 'models', 'HillClimbModel3', 'model_PPO_1_new'))

episodes = 20
for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(state)
        action = int(action)
        state, reward, done, _, _ = env.step(action=action)
        score += reward
    print('Episodes: {}, Score: {}'.format(episode, score))