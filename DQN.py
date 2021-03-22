import numpy as np
import matplotlib.pyplot as plt
import gym
from IPython.display import display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72.0)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('movie_cartpole.mp4')
    display(display_animation(anim, default_mode='loop'))


ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 500
CAPACITY = 1000
BATCH_SIZE = 32

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # �൱�ڶ�����һ��Transition��

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # ����Brain�࣬Brain��ʵ����

    def update_Q_function(self, observation, action, reward, observation_next):
        """
        ִ��update_Q_table
        """
        self.brain.replay()

    def get_action(self, state, step):
        """
        ִ��action"""
        action = self.brain.decide_action(state, step)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = False
        frames = []
        is_episode_final = False
        for episode in range(NUM_EPISODES):

            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            """state��ֵ+��ʽת��"""

            for step in range(MAX_STEPS):
                if is_episode_final is True:  # ִ�����һ��ѭ������״̬��������frames
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_actiosn(observation, episode)  # ����
                observation_next, _, done, _ = self.env.step(action.item())  # ִ��action�������ض������״̬

                if done:
                    step_next = None
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes += 1
                else:
                    reward = torch.FloatStorage([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)
                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_Q_function()
                state = state_next
                if done:
                    print('{0} Episode:Finished after {1} time steps'.format(episode, step + 1))
                    break

            if is_episode_final is True:
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('10�غ������ɹ�')
                is_episode_final = True


class ReplayMemory:  # �洢����
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity
        """����capacity��������capacityһ������memory�ĳ���"""

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # �����ȡmemorys���������batch_size��

    def __len__(self):
        return len(self.memory)


class Brain:  # ���ݾ��飬�������ߣ���Brain���д������

    def __init__(self, num_states, num_actions):  # ��ʼ��������
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))  # ����num_state�����Ϊ32��ȫ���Ӳ�
        self.model.add_module('relu1', nn.ReLU())  # �����relu
        self.model.add_module('fc2', nn.Linear(32, 32))  # 32��32��ȫ���Ӳ�
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # �Ż�model�Ĳ���

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)  # ����replaymemory�ִ࣬��������������б�
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)  # ��Ϊtorch������size = (batch_size,1)
        action_batch = torch.cat(batch.action)  #
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # s+1״̬��������Сδ֪
        """next_stateת��Ϊ����"""

        self.model.eval()  # Ӧ��ģʽ
        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)  # tensor ��ֵΪ0�����ȵ���batch_size
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values
        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))  # ��ʧ����
        """
        state_action_values
        expected_state_action_values:�����Ķ���  
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)  # ����reshape?
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

            return action
