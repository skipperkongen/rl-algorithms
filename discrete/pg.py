"""
Credit:
- Policy Gradients Are Easy In Keras: https://www.youtube.com/watch?v=IS0V8z8HXrM
- How Policy Gradient Reinforcement Learning Works: https://youtu.be/A_2U6Sx67sE
- https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146
"""

from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

class PGAgent():
    def __init__(self, lr, gamma=0.99, n_actions=4,
                 h1_size=16, h2_size=16, input_dims=(128,),
                 fname='reinforce.h5'):
        self.lr = lr
        self.gamma = gamma
        self.G = 0
        self.n_actions = n_actions
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.input_dims = input_dims
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(n_actions)]

        self.model_file = fname

    def build_policy_network(self):
        input = Input(shape=self.input_dims)
        advantages = Input(shape=[1])
        dense1 = Dense(self.h1_size, activation='relu')(input)
        dense2 = Dense(self.h2_size, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        policy = Model(inputs=[input, advantages], outputs=[probs])

        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

        predict = Model(inputs=[input], outputs=[probs])

        return policy, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std

        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return cost

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)        
