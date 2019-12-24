import gym
import numpy as np
from discrete.dqn import DQNAgent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_episodes = 500
    agent = DQNAgent(
        lr=5e-4, gamma=0.99, epsilon=0.0, epsilon_dec=0.996,
        epsilon_end=0.0, input_dims=env.observation_space.shape,
        n_actions=env.action_space.n, h1_size=256, h2_size=256,
        mem_size=int(1e6), batch_size=64,
    )

    # agent.load_model()
    scores = []
    eps_history = []

    for i in range(num_episodes):
        done = False
        score = 0
        state = env.reset()
        while not done:
            if i % 10 == 0:
                env.render()
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            agent.remember(state, action, reward, state_, int(done))
            state = state_
            _ = agent.learn()

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: {} score: {:.1f} avg. score: {:.1f} epsilon: {:.2f}'.format(
            i, score, avg_score, agent.epsilon
        ))











# foo
