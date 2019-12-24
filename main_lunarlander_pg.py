import gym
import numpy as np
from discrete.pg import PGAgent

if __name__=='__main__':
    env = gym.make('LunarLander-v2')
    print()
    agent = PGAgent(
        lr=5e-4,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        h1_size=64,
        h2_size=64
    )

    score_history = []
    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            if i % 100 == 0:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        _ = agent.learn()
        print('episode: ', i,'score: %.1f' % score,
            'average score %.1f' % np.mean(score_history[max(0, i-100):(i+1)]))
