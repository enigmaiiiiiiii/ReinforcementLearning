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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # 相当于定义了一个Transition类

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # 创建Brain类，Brain类实例化

    def update_Q_function(self, observation, action, reward, observation_next):
        """
        执行update_Q_table
        """
        self.brain.replay()

    def get_action(self, state, step):
        """
        执行action"""
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
            """state赋值+格式转换"""

            for step in range(MAX_STEPS):
                if is_episode_final is True:  # 执行最后一次循环，将状态动画加入frames
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_actiosn(observation, episode)  # 动作
                observation_next, _, done, _ = self.env.step(action.item())  # 执行action，并返回动作后的状态

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
                print('10回合连续成功')
                is_episode_final = True


class ReplayMemory:  # 存储经验
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity
        """除以capacity的余数，capacity一档大于memory的长度"""

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 随机抽取memorys储存的样本batch_size个

    def __len__(self):
        return len(self.memory)


class Brain:  # 根据经验，做出决策，在Brain体中搭建神经网络

    def __init__(self, num_states, num_actions):  # 初始化神经网络
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))  # 输入num_state，输出为32的全连接层
        self.model.add_module('relu1', nn.ReLU())  # 激活函数relu
        self.model.add_module('fc2', nn.Linear(32, 32))  # 32，32的全连接层
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # 优化model的参数

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)  # 调用replaymemory类，执行随机抽样返回列表
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)  # 变为torch张量，size = (batch_size,1)
        action_batch = torch.cat(batch.action)  #
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # s+1状态张量，大小未知
        """next_state转换为张量"""

        self.model.eval()  # 应用模式
        state_action_values = self.model(state_batch).gather(1, action_batch)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)  # tensor 数值为0，长度等于batch_size
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values
        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))  # 损失函数
        """
        state_action_values
        expected_state_action_values:期望的动作  
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)  # 类似reshape?
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

            return action
