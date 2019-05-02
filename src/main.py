#!/usr/bin/python3
import gym
import numpy as np
import signal
from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer
import matplotlib.pyplot as plt
import reco_gym
from collections import namedtuple
import random
from reco_gym import env_1_args, Configuration

def run(episodes=100,
        render=False,
        max_actions=1e6,
        knn=0.1):
    experiment = 'reco-gym-v1'
    env = gym.make(experiment)
    env_1_args['num_products'] = 10
    env.init_gym(env_1_args)
    obs_space = namedtuple('obs_space', ['n'])
    env.observation_space = obs_space(env_1_args['num_products'] * 2)
    print(env.observation_space)
    print(env.action_space)

    steps = 10 #env.spec.timestep_limit

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn, dim_embed=2)

    timer = Timer()

    data = util.data.Data()
    data.set_agent(agent.get_name(), int(agent.action_space.get_number_of_actions()),
                   agent.k_nearest_neighbors, 3)
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)

    agent.add_data_fetch(data)
    print(data.get_file_name())
    episode_total = 0
    cumulative_sum = 0

    full_epoch_timer = Timer()
    rewards = []

    def plot(to=None):
        plt.figure()
        plt.plot(np.convolve(rewards, [10 / len(rewards)] * (len(rewards) // 10))[len(rewards) // 10 : -len(rewards) // 10])
        if to:
            plt.savefig(to)
        plt.show()

    def parse(obs):
        img = np.zeros(env.observation_space.n)
        if obs:
            print(obs.sessions())
            for i in obs.sessions():
                img[i['v']] += 1
                img[i['u'] + env.observation_space.n // 2] += 1
        return img

    def train(episodes):
        stop = False
        def handler(a, b):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGINT, handler)

        nonlocal episode_total
        nonlocal cumulative_sum
        curr_tot = episode_total

        for ep in range(episodes):
            if stop: break
            timer.reset()
            observation = parse(env.reset())
            observation, reward, done, info = env.step(None)
            observation = parse(observation)
            episode_total += 1
            total_reward = 0
            print('Episode ', episode_total, 'started...', end='')
            agent.make_embed()
            for t in range(steps):

                if render:
                    env.render()

                action = agent.act(observation)
                if random.random() < 0.1: print(action)

                data.set_action(action.tolist())

                data.set_state(observation.tolist())

                prev_observation = observation
                observation, reward, done, info = env.step(int(action[0] if len(action) == 1 else action))
                observation = parse(observation)
                data.set_reward(reward)

                episode = {'obs': prev_observation,
                           'action': action,
                           'reward': reward,
                           'obs2': observation,
                           'done': done,
                           't': t}

                agent.observe(episode)
                total_reward += reward
                if done or (t == steps - 1):
                    t += 1
                    cumulative_sum += total_reward
                    rewards.append(total_reward)
                    time_passed = timer.get_time()
                    print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
                                                                                time_passed, round(
                                                                                    time_passed / t),
                                                                                round(cumulative_sum / episode_total)))

                    #data.finish_and_store_episode()

                    break
        # end of episodes
        time = full_epoch_timer.get_time()
        print('Run {} episodes in {} seconds and got {} average reward'.format(
            episode_total - curr_tot, time / 1000, cumulative_sum / episode_total))
        #data.save()
        signal.signal(signal.SIGINT, signal.SIG_DFL)


    def iterate(num=20):
        for i in range(num): 
            agent.unfreeze()
            train(200)
            agent.freeze()
            train(1000)

    from IPython import embed; embed()


if __name__ == '__main__':
    run()
