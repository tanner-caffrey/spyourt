# uh oh
# that's a bad name let's hope it changes

import os
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory,time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.drivers import py_driver, dynamic_episode_driver
from tf_agents.utils import common
from tf_agents.policies import PolicySaver
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime

class SimpleGridWorldEnv(py_environment.PyEnvironment):
    def __init__(self, shape, ):
        self.shape = shape
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=shape, dtype=np.int32, minimum=[0,0,0,0,0,0],maximum=[5,5,5,5,5,5], name='observation')
        s1 = rand.randint(0,5)
        s2 = rand.randint(0,5)
        self._state=[s1,s2,rand.randint(0,5),rand.randint(0,5),s1,s2]
        self._episode_ended = False
        self.initial_start = (self._state[0],self._state[1])

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def get_state(self):
        return self._state

    def _reset(self):
        s1 = rand.randint(0,5)
        s2 = rand.randint(0,5)
        self._state=[s1,s2,rand.randint(0,5),rand.randint(0,5),s1,s2]        
        self.initial_start = (self._state[0],self._state[1])
        if self.game_over(): return self._reset()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))
    
    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self.move(action)

        if self.game_over():
            self._episode_ended = True

        if self._episode_ended:
            if self.game_over():
                reward = 100
            else:
                reward = self.get_reward()[0]
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0, discount=0.9)
    
    def move(self, action):
        row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
        if action == 0: #down
            if row - 1 >= 0:
                self._state[0] -= 1
        if action == 1: #up
            if row + 1 < 6:
                self._state[0] += 1
        if action == 2: #left
            if col - 1 >= 0:
                self._state[1] -= 1
        if action == 3: #right
            if col + 1  < 6:
                self._state[1] += 1

    def get_reward(self):
        start = np.array(self.initial_start)
        end = np.array((self._state[2],self._state[3]))
        cur = np.array((self._state[0],self._state[1]))
        init_dist = np.sum(np.abs(start-end))
        cur_dist = np.sum(np.abs(cur-end))
        dif = init_dist-cur_dist
        reward = dif/2 #if dif>0 else 0
        return (reward, dif)

    def game_over(self):
        row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
        return row==frow and col==fcol
