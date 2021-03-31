import torch.nn.functional as F
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
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import yaml

with open('cfg.yaml', encoding="gbk") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

ENV = cfg['ENV']
GAMMA = cfg['GAMMA']
MAX_STEPS = cfg['MAX_STEPS']
NUM_EPISODES = cfg['NUM_EPISODES']
CAPACITY = cfg['CAPACITY']
BATCH_SIZE = cfg['BATCH_SIZE']


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72.0)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('movie_cartpole.mp4')
    display(display_animation(anim, default_mode='loop'))


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # 相当于定义了一个Transition类 一个Q(s,a)网络


class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        output = self.fc3(h2)

        return output


class ReplayMemory:  # 存储经验
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # memory元素+1
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity
        """除以capacity的余数，capacity一直大于memory的长度"""

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 随机抽取memory储存的样本batch_size（批次）个用作训练

    def __len__(self):
        return len(self.memory)


class Brain:  # 根据经验，做出决策，在Brain体中搭建神经网络
    """ model is the model 包含__init__和forward方法"""

    def __init__(self, num_states, num_actions):  # 初始化神经网络
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out = num_states, 32, num_actions
        """中间层为32的全连接层，输入层维度=state size，输出层维度=action size"""
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)  # 深度学习网络
        self.loss = 0

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)  # 优化model的参数

    def replay(self):
        """在记忆中训练"""

        # 确认存储器大小
        if len(self.memory) < BATCH_SIZE:
            return

        # 小批量的制作
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 求期望的Q(s_t, a_t)值
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 耦合参数的更新
        self.update_main_q_network()

    def decide_action(self, state, episode):

        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
                """从main_q_network中返回动作"""
        else:

            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self):

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):

        self.main_q_network.eval()
        self.target_q_network.eval()
        """
        求网络输出的Q(s_t, a_t)
        self.model(state_batch)输出左右两边的Q值，
        变成了[torch.FloatTensor of size batch_size × 2]。
        为了求出与从这里执行的动作a_t相对应的Q值，需要在action_batch中执行的动作a_t是右还是左的index
        用gather推导出对应的Q值。
        """
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        """
        max{Q(s_t+1, a)}求值。但是要注意是否有以下状态。
        54/5000 创建索引掩码，检查cartpole是否为done，是否存在next_state。
        """
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))  # map(function,*iterables)

        # 首先全部都是0
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 从Main Q- network中求出下一状态下最大Q值的行动a_m
        # 最后的[1]返回对应于行为的index
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 只过滤有以下状态的物体，size 32变为32×1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        """
        从target Q- network中获取具有下一状态的index的行为a_m的Q值
        detach()取出
        squeeze()将size[minibatch×1]变成[minibatch]。
        """

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # Q(s_t, a_t)值由Q学习式求出
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()  # 重置梯度
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):  # DDQN新增方法

        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # 创建Brain类，Brain类实例化

    def update_Q_function(self):
        """
        执行update_Q_table
        """
        self.brain.replay()

    def get_action(self, state, step):
        """
        执行action
        """
        action = self.brain.decide_action(state, step)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        """更新Target Q-Network与Main Q-Network相同"""
        self.brain.update_target_q_network()


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
        for episode in range(NUM_EPISODES):

            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            steps = 0
            """state赋值+格式转换"""

            for step in range(MAX_STEPS):
                if episode_final is True:  # 执行最后一次循环，将状态动画加入frames
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(state, episode)  # 动作
                observation_next, _, done, _ = self.env.step(action.item())  # 执行action，并返回动作后的状态

                if done:
                    state_next = None
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                    if step < 195:  # 如果半途倒下，reward=-1
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0  # 重置连续成功记录
                    else:
                        reward = torch.FloatTensor([1.0])  # 站立200步，获得奖励 reward = 1
                        complete_episodes += 1
                else:
                    reward = torch.FloatTensor([0.0])

                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_Q_function()
                state = state_next
                if done:
                    steps = step + 1
                    print('{0} Episode:Finished after {1} times steps'.format(episode, steps))
                    break

            writer.add_scalar("episodeloss", self.agent.brain.loss, episode)
            # writer.add_scalar("loss", self.agent.brain.loss, episode)
            if episode_final is True:
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('10回合连续成功')
                episode_final = True


writer = SummaryWriter("./logs")
cartpole_env = Environment()
cartpole_env.run()
