#!/usr/bin/python3
import gym
import numpy as np
import signal
from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer
import matplotlib.pyplot as plt
#import reco_gym
from collections import namedtuple
import random
#from reco_gym import env_1_args, Configuration

def run(episodes=100,
        render=False,
        max_actions=1e3,
        experiment='InvertedPendulum-v2',
        iters=20,
        name=None,
        dim=3,
        knn=1.0):
    env = gym.make(experiment)
    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn, dim_embed=dim)

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
        plt.title(experiment)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        n = len(rewards) // 20
        r = rewards
        smooth = np.convolve(n * [r[0]] + rewards + n * [r[-1]],
            [1 / n] * n)
        c = plt.plot(r, alpha=0.2)
        plt.plot(smooth[n : len(r) + n], c=c[0].get_color())
        if to:
            plt.savefig(to)

    def parse(obs):
        return obs

    lock = False

    def train(episodes):
        stop = False
        def handler(a, b):
            nonlocal stop
            nonlocal lock
            lock = True
            stop = True
        signal.signal(signal.SIGINT, handler)

        nonlocal episode_total
        nonlocal cumulative_sum
        curr_tot = episode_total

        for ep in range(episodes):
            if stop: break
            timer.reset()
            observation = parse(env.reset())
            episode_total += 1
            total_reward = 0
            print('Episode ', episode_total, 'started...', end='')
            agent.make_embed()
            for t in range(steps):

                if render:
                    env.render()

                action = agent.act(observation)

                data.set_action(action.tolist())

                data.set_state(observation.tolist())

                prev_observation = observation
                observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)
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


    def iterate(num=iters):
        nonlocal lock
        nonlocal knn
        for i in range(num): 
            agent.train_agent()
            train(100)
            if lock: break
            agent.train_embed()
            train(20)
            if lock: break
            knn /= 2
            if knn < 0.1:
                knn = 0.1
            agent.anneal(knn)
            plot(name)
        lock = False
    
    iterate()


if __name__ == '__main__':
    from IPython import embed; embed(using=0)
