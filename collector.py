import gym
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
try:
    import gym_snake
except ImportError:
    print("You have not installed gym-snake! Do not try to use env_type=\'snake-v0\'")


class Collector():
    """
    This class handles the collection of data by interacting with the environments.
    """

    if torch.cuda.is_available():
        torch.FloatTensor = torch.cuda.FloatTensor
        torch.LongTensor = torch.cuda.LongTensor

    def __init__(self, reward_q, grid_size=[20,20], n_snakes=2, n_foods=4, unit_size=4, n_obs_stack=2, net=None, n_tsteps=15, gamma=0.995, env_type='snake-plural-v0', preprocessor= lambda x: x):

        self.preprocess = preprocessor
        self.env_type = env_type
        self.env = gym.make(env_type)
        self.env.grid_size = grid_size
        self.env.n_foods = n_foods
        self.env.unit_size = unit_size
        self.env.n_snakes = n_snakes
        self.action_space = self.env.action_space.n

        observation = self.env.reset()
        snake_views = self.get_snake_views(observation)
        self.obs_shape = observation.shape
        self.prepped_shape = snake_views[0].shape
        self.state_shape = [n_obs_stack*self.prepped_shape[0],*self.prepped_shape[1:]]
        self.snake_states = []
        for i in range(self.env.n_snakes):
            self.snake_states.append(self.make_state(snake_views[i]))

        self.gamma = gamma
        self.net = net
        self.n_tsteps = n_tsteps
        self.reward_q = reward_q
        self.alive = np.ones(n_snakes, dtype=np.int32)

    def produce_data(self, data_q):
        """
        Used as the external call to get a rollout from each environment.

        Adds a tuple of data from a rollout to the process queue.
        data_q - multiprocessing.Queue that stores data to train the policy.
        """

        self.net.req_grads(False)
        self.net.train(mode=False)
        while True:
            data = self.rollout()
            data_q.put(data)

    def render(self, render):
        """
        Used to watch the environment unfold using the collector's policy.
        """
        return self.rollout(render=render)

    def get_vid_data(self, min_reward=7):
        """
        Continuously rolls out the environment until the minimum reward is achieved.
        The sequence of observations is then returned to make a video.
        """

        total_reward = 0
        vid_imgs = []
        quota_reached = False
        done = False
        while not (quota_reached and done):
            if done:
                total_reward = 0
                vid_imgs = []

            numpy_states = np.asarray(self.snake_states, dtype=np.float32)
            state = torch.FloatTensor(numpy_states)
            values, pis = self.net.forward(Variable(state))
            actions = []
            for j in range(pis.shape[0]):
                action = self.get_action(pis[j].data)
                actions.append(action)

            obs, rewards, done, info = self.env.step(actions)
            vid_imgs.append(obs)

            for j,reward in enumerate(rewards):
                if reward == -1:
                    self.alive[j] = 0
                    done = True
                else:
                    total_reward += reward
                    if total_reward >= min_reward:
                        quota_reached = True

            if done:
                obs, self.snake_states = self.reset()
                print("Episode Reward:", total_reward)

            self.snake_states = self.next_states(self.snake_states, obs)

        return vid_imgs


    def rollout(self, render=False):
        """
        Collects a rollout of n_tsteps in the given environment. The collected data
        are the states that were used to get the actions, the actions that
        were used to progress the environment, the rewards that were collected from
        the environment, and the done signals from the environment.

        Returns python lists of the relavent data.
        states - python list of all states collected in this rollout
        rewards - python list of float values collected from rolling out the environments
        dones - python list of booleans denoting the end of an episode
        actions - python list of integers denoting the actual selected actions in the
                    rollouts
        advantages - python list of floats denoting the td value error corresponding to
                    the equation r(t) + gamma*V(t+1) - V(t)
        """

        ep_states, ep_rewards, ep_dones, ep_actions, ep_advantages = [], [], [], [], []
        for i in range(self.n_tsteps):
            if render: self.env.render(mode='human')
            numpy_states = np.asarray(self.snake_states, dtype=np.float32)
            state = torch.FloatTensor(numpy_states)
            values, pis = self.net.forward(Variable(state))
            actions = []
            for j in range(pis.shape[0]):
                action = self.get_action(pis[j].data)
                actions.append(action)

            obs, rewards, done, info = self.env.step(actions)

            for j,reward in enumerate(rewards):
                if reward != 0:
                    self.reward_q.put(.99*self.reward_q.get() + .01*reward)
                    if reward == -1:
                        self.alive[j] = 0
                        done = True

            values = values.data.squeeze().numpy()*self.alive
            rewards = np.array(rewards, dtype=np.float32)

            ep_states.append(self.snake_states.copy())
            if done:
                obs, self.snake_states = self.reset()

            if i > 0:
                keep_val = (1-done)
                advantages = ep_rewards[-1] + self.gamma*values*keep_val - last_values
                ep_advantages.append(advantages)
            last_values = values

            ep_rewards.append(rewards), ep_dones.append(1-self.alive), ep_actions.append(actions)
            self.snake_states = self.next_states(self.snake_states, obs)

        ep_rewards[-1] = ep_rewards[-1] + last_values # Bootstrapped value
        ep_dones[-1] = np.ones(self.alive.shape, dtype=np.int32)
        ep_advantages.append(ep_rewards[-1]-last_values)
        data = ep_states, ep_rewards, ep_dones, ep_actions, ep_advantages
        data = self.unroll(data)
        return data

    def unroll(self, data):
        new_data = []
        for i, d in enumerate(data):
            npd = np.asarray(d)
            if i == 0: # States treated differently
                npd = npd.transpose((1,0,*list(range(2,len(npd.shape))))).reshape((-1, *npd.shape[2:]))
                new_data.append(npd)
            else:
                new_data.append(npd.T.ravel())
        return new_data

    def get_action(self, pi):
        """
        Stochastically selects an action based on the action probabilities.

        pi - torch FloatTensor of the raw action prediction
        """

        action_ps = self.softmax(pi.numpy()).squeeze()
        action = np.random.choice(self.action_space, p=action_ps)
        return int(action)

    def get_snake_views(self, obs):
        """
        Convert the observation to a common color scheme for each snake.
        """
        snake_views = []
        for snake_idx in range(self.env.n_snakes):
            snake_view = self.snake_view_prep(obs, snake_idx)
            snake_views.append(snake_view)
        return snake_views

    def make_state(self, prepped_obs, prev_state=None):
        """
        Combines the new, prepprocessed observation with the appropriate parts of the previous
        state to make a new state that is ready to be passed through the network. If prev_state
        is None, the state is filled with zeros outside of the new observation.

        prepped_obs - torch FloatTensor of peprocessed observation
        prev_state - torch FloatTensor of previous environment state
        """

        if prev_state is None:
            prev_state = np.zeros(self.state_shape, dtype=np.float32)

        next_state = np.concatenate([prepped_obs, prev_state[:-prepped_obs.shape[0]]], axis=0)
        return next_state

    def next_states(self, prev_states, observation):
        """
        Get the next state of the environment.

        env - environment of interest
        prev_state - ndarray of the state used in the most recent action
                    prediction
        obs - ndarray returned from the most recent step of the environment
        """
        snake_views = self.get_snake_views(observation)
        new_states = []
        for i in range(len(prev_states)):
            new_states.append(self.make_state(snake_views[i], prev_states[i]))
        return new_states

    def preprocess(self, pic):
        """
        Each raw observation from the environment is run through this function.
        Put anything sort of preprocessing into this function.
        This function is set in the intializer.

        pic - ndarray of an observation from the environment [H,W,C]
        """
        pass

    def reset(self):
        """
        Resets environment and corresponding variables.
        """
        obs = self.env.reset()
        new_states = []
        for i in range(len(self.snake_states)):
            new_states.append(None)
            self.alive[i] = True
        return obs, new_states


    def snake_view_prep(self, obs, snk_idx):
        if self.env.controller.snakes[snk_idx] is not None:
            head_color = self.env.controller.snakes[snk_idx].head_color
            view = self.preprocess(obs, head_color)
        else:
            view = np.zeros(self.prepped_shape)
        return view

    def softmax(self, X, theta=1.0, axis=-1):
        """
        * Inspired by https://nolanbconaway.github.io/blog/2017/softmax-numpy *

        Computes the softmax of each element along an axis of X.

        X - ndarray of at least 2 dimensions
        theta - float used as a multiplier prior to exponentiation
        axis - axis to compute values along

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

        X = X * float(theta)
        X = X - np.expand_dims(np.max(X, axis = axis), axis)
        X = np.exp(X)
        ax_sum = np.expand_dims(np.sum(X, axis = axis), axis)
        p = X / ax_sum
        return p
