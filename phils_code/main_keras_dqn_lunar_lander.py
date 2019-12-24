from simple_dqn_keras import Agent
import numpy as np
import gym
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_episodes = 500
    agent = Agent(
        gamma=0.99, epsilon=0.0,
        alpha=5e-4, input_dims=8,
        n_actions=4, mem_size=1000000,
        batch_size=64, epsilon_end=0.0
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
#        if i % 10 == 0 and i > 0:
#            agent.save_model()

    filename = 'lunarlander.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)






# foo
