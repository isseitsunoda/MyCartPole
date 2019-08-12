import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.losses import huber_loss
from gym import wrappers

EPISODES = 500
ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEP = 200
NUM_EPISODES = 500

CAPACITY = 100000
BATCH_SIZE = 32

class ExperienceMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Q_network:
    def __init__(self, num_state, num_action):
        self.memory = ExperienceMemory(CAPACITY)
        self.num_state, self.num_action = num_state, num_action

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(num_state, )))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(num_action, activation='linear'))
        self.optimizer = Adam(lr=0.00001)
        self.model.compile(loss=huber_loss, optimizer=self.optimizer)

    def replay(self, target_q_n):
        if len(self.memory) < BATCH_SIZE:
            return
        inputs = np.zeros((BATCH_SIZE, self.num_state))
        targets = np.zeros((BATCH_SIZE, self.num_action))
        transitions = self.memory.sample(BATCH_SIZE)

        for i, (state_batch, action_batch, next_state_batch, reward_batch) in enumerate(transitions):
            inputs[i:i+1] = state_batch
            #print(inputs[i:i + 1], '%%%', inputs[i])
            target = reward_batch

            if not (next_state_batch == np.zeros(state_batch.shape)).all(axis=1):
                mainQ = self.model.predict(state_batch)[0]
                next_action = np.argmax(mainQ)
                target = reward_batch + GAMMA * target_q_n.model.predict(next_state_batch)[0][next_action]

            targets[i] = self.model.predict(state_batch)
            targets[i][action_batch] = target
            self.model.fit(inputs, targets, epochs=1, verbose=0)

    def decide_action(self, state, episode, target_q_n):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            a = target_q_n.model.predict(state)
            A = np.argmax(a[0])
            #print(a, A, a[0], sep='###')
            action = A
        else:
            action = np.random.choice(self.num_action)

        return action


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Q_network(num_states, num_actions)
        self.target_q_n = Q_network(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay(self.target_q_n)

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode, self.target_q_n)

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        """self.env = wrappers.Monitor(self.env, './movie/cartpoleDQN', force=True,
                                    video_callable=(lambda epi: epi % 10 == 0))  # 10回ごとにビデオを保存　"""
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        complete_episodes = 0
        episode_final = False

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            state = observation
            state = np.reshape(state, [1, self.num_states])

            self.agent.target_q_n = self.agent.brain

            for step in range(MAX_STEP):
                action = self.agent.get_action(state, episode)
                next_state, _, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.num_states])

                if done:
                    next_state = np.zeros(state.shape)
                    if step < 195:
                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes += 1
                else:
                    reward = 0

                self.agent.memorize(state, action, next_state, reward)
                self.agent.update_q_function()

                state = next_state

                if done:
                    print('{} Episode: Finished after {} steps: complete_episodes: {}'.format(
                        episode, step+1, complete_episodes))
                    break

                if episode_final:
                    self.env.render()
                    break

                if complete_episodes >= 10:
                    print("10回連続成功")
                    episode_final = True


if __name__ == '__main__':
    cartpole_env = Environment()
    cartpole_env.run()
