import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from matplotlib import cm
import random

"""
θ中包含状态和选择
策略迭代法：new_theta,new_pi
简直迭代：new_Q
"""


def simple_convert_into_pi_from_theta(theta):
    """
返回策略集合PI，各个状态的各个选择的概率，直接转换,既每一步策略随机选择，且大小不变，
    """
    beta = 0.1  #
    [m, n] = theta.shape
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta)  # e^(β*θ)
    for i in range(m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)  # 矩阵中的空值转换为数字0

    return pi


def get_s_next(s, a):  # 返回动作a后的状态
    # a = get_action()
    direction = ['up', 'right', 'down', 'left']
    next_direction = direction[a]

    if next_direction == 'up':
        s_next = s - 3  # s'
    elif next_direction == 'right':
        s_next = s + 1  # s'
    elif next_direction == 'down':
        s_next = s + 3  # s'
    elif next_direction == 'left':
        s_next = s - 1  # s'

    return s_next


def get_action(s, Q, epsilon, pi_0):  # 有一定概率选择最大动作价值行动，返回action
    """
    s是当前状态
    np.nanargmax[]返回[]内最大值的索引
    """

    direction = ['up', 'right', 'down', 'left']
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]  # 返回nan之外最大值的索引

    if next_direction == 'up':
        action = 0  # a
    elif next_direction == 'right':
        action = 1  # a
    elif next_direction == 'down':
        action = 2  # a
    elif next_direction == 'left':
        action = 3  # a
    return action


def goal_maze_ret_s_a_q(Q, epsilon, eta, gamma, pi):  # 返回路径和价值
    """
    完成1次任务
    """
    s = 0
    a_next = get_action(s, Q, epsilon, pi)
    s_a_history = [[0, np.nan]]

    while (1):

        """
        状态等于8时完成任务退出循环，返回路径日志和价值函数
        """
        a = a_next  # 更新a
        s_a_history[-1][1] = a  # [[s1,a1],[s2,a2],...
        s_next = get_s_next(s, a)
        s_a_history.append([s_next, np.nan])  # 下一个状态，和下一个动作
        if s_next == 8:  # 状态8完成，reward = 1
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)

        Q = update_q(s, a, r, s_next, a_next, eta, gamma, Q)
        """每次动作都会更新Q"""

        if s_next == 8:
            break
        else:
            s = s_next
    return [s_a_history, Q]


def update_q(s, a, r, s_next, a_next, eta, gamma, Q):  # 价值Q更新，迭代函数
    """
    eta:
    gamma:
    """
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])  # Q学习
        # Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])  # Sarsa
    if s == 4:
        print("当下一个状态是{}时\t".format(s_next),
              "期望Q值:{:4f}".format(r + gamma * Q[s_next, a_next])+"\t",
              "当前Q值:{:4f}".format(Q[s,a]))

    return Q


theta_0 = np.array(
    [[np.nan, 1, 1, np.nan],  # 方案，用于转换为概率的值
     [np.nan, 1, np.nan, 1],
     [np.nan, np.nan, 1, 1],
     [1, 1, 1, np.nan],
     [np.nan, np.nan, 1, 1],
     [1, np.nan, np.nan, np.nan],
     [1, np.nan, np.nan, np.nan],
     [1, 1, np.nan, np.nan]])
pi_0 = simple_convert_into_pi_from_theta(theta_0)
a, b = theta_0.shape
Q = np.random.rand(a, b) * theta_0 * 0.1  # 价值函数集合，是一个矩阵
eta = 0.1  # 学习率
gamma = 0.9
epsilon = 0.5
is_continue = True
episode = 1
v = np.nanmax(Q, axis=1)

while is_continue:
    """
    只迭代Q：价值表
    """
    print("回合数：" + str(episode)+"-----------------------------")
    epsilon = epsilon / 2
    s_a_history, Q = goal_maze_ret_s_a_q(Q, epsilon, 0.3, 0.1, pi_0)
    new_v = np.nanmax(Q, axis=1)  # 每个状态价值最大值
    v = new_v
    print("求解迷宫为题所需的步数是：" + str(len(s_a_history) - 1))
    episode = episode + 1
    if episode > 20:
        break
