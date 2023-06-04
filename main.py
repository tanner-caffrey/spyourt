# uh oh
# that's a bad name let's hope it changes

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
import matplotlib.pyplot as plt
import random as rand

LEARNING_RATE = 1

class GridWorldEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.int32, minimum=[0,0,0,0,0,0],maximum=[5,5,5,5,5,5], name='observation')
        s1 = rand.randint(0,5)
        s2 = rand.randint(0,5)
        self._state=[s1,s2,rand.randint(0,5),rand.randint(0,5),s1,s2] #represent the (row, col, frow, fcol) of the player and the finish
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

def update_rendering(base_grid, state, agent='Q', target='X', object='o'):
    grid = base_grid.copy()
    if grid[state[0]][state[1]] == target:
        grid[state[0]][state[1]] = '!'
        return grid
    elif grid[state[0]][state[1]] == agent:
        grid[state[0]][state[1]] = 'O'
    elif grid[state[0]][state[1]] == '@':
        pass
    else:
        grid[state[0]][state[1]] = agent
    grid[state[2]][state[3]] = target
    return grid

def render_progress(environment, policy):
    env = environment.pyenv.envs[0]
    time_step = environment.reset()
    grid = np.full((6,6),'~')
    grid = update_rendering(grid,env.get_state(),agent='@')
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        grid = update_rendering(grid,env.get_state())
    print(grid)
    print(env.get_reward())

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0    
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

# parameter settings
num_iterations = 100000  
initial_collect_steps = 1000  
collect_steps_per_iteration = 1  
replay_buffer_capacity = 100000  
fc_layer_params = (100,45)
batch_size = 128 # 
learning_rate = 1e-5  
log_interval = 200   
num_eval_episodes = 2
eval_interval = 1000  

train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn = common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

tf_agent.initialize()
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = tf_agent.collect_data_spec,
        batch_size = train_env.batch_size,
        max_length = replay_buffer_capacity)
print("Batch Size: {}".format(train_env.batch_size))
replay_observer = [replay_buffer.add_batch]
train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
]

dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=replay_observer + train_metrics,
    num_steps=1)
iterator = iter(dataset)
print(compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes))
tf_agent.train = common.function(tf_agent.train)
tf_agent.train_step_counter.assign(0)
final_time_step, policy_state = driver.run()

episode_len = []
step_len = []
for i in range(num_iterations):
    final_time_step, _ = driver.run(final_time_step, policy_state)
    experience, _ = next(iterator)
    train_loss = tf_agent.train(experience=experience)
    step = tf_agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        episode_len.append(train_metrics[3].result().numpy())
        step_len.append(step)
        print('Average episode length: {}'.format(train_metrics[3].result().numpy()))
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy,num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
    if step % 350 == 0:
        render_progress(eval_env, tf_agent.policy)

plt.plot(step_len, episode_len)
plt.xlabel('Episodes')
plt.ylabel('Average Episode Length (Steps)')
plt.show()
    
# if __name__ == '__main__':
#     env = GridWorldEnv()
