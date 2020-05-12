from rl.callbacks import Callback
import timeit
from keras.utils.generic_utils import Progbar
import numpy as np
from keras import __version__ as KERAS_VERSION
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TrainIntervalLogger, Callback
import matplotlib.pyplot as plt
import os
import csv
import glob
import shutil

class CumulativeRewardLogger(TrainIntervalLogger):
    def __init__(self, env, filename, interval=20):
        TrainIntervalLogger.__init__(self, interval)
        self.cumulative_reward = 0
        self.filename = filename
        self.data = []
        self.env = env
    def on_train_begin(self, logs):
        """ Initialize training statistics at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
    def on_episode_begin(self, episode, logs):
        """ Initialize metrics at the beginning of each episode """
        print("------------------ new episode begin, environment reset -----------------------------")
    def on_step_end(self, step, logs):
        """ Update progression bar at the end of each step """
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        self.cumulative_reward = self.env._total_reward
        values = [('reward', logs['reward']), ('cumulative_reward', self.cumulative_reward)]
        self.data.append([step, logs['reward'], self.cumulative_reward])
        if KERAS_VERSION > '2.1.3':
            self.progbar.update((self.step % self.interval) + 1, values=values)
        else:
            self.progbar.update((self.step % self.interval) + 1, values=values, force=True)
        self.step += 1
        self.metrics.append(logs['metrics'])
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])
    def on_train_end(self, logs):
        np_data = np.array(self.data)
        np.savetxt(self.filename, np_data)
class TestCallback(TrainIntervalLogger):
    def __init__(self, env, filename, interval=20):
        TrainIntervalLogger.__init__(self, interval)
        self.cumulative_reward = 0
        self.filename = filename
        self.data = []
        self.env = env
    def on_train_begin(self, logs):
        """ Initialize training statistics at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
    def on_episode_begin(self, episode, logs):
        """ Initialize metrics at the beginning of each episode """
        print("------------------ new episode begin, environment reset -----------------------------")
    def on_step_end(self, step, logs):
        """ Update progression bar at the end of each step """
        self.cumulative_reward = self.env._total_reward
        values = [('reward', logs['reward']), ('cumulative_reward', self.cumulative_reward)]
        self.data.append([step, logs['reward'], self.cumulative_reward])
        if KERAS_VERSION > '2.1.3':
            self.progbar.update((self.step % self.interval) + 1, values=values)
        else:
            self.progbar.update((self.step % self.interval) + 1, values=values, force=True)
        self.step += 1

    def on_train_end(self, logs):
        np_data = np.array(self.data)
        np.savetxt(self.filename, np_data)
class Evaluator:
    def __init__(self, agent_creator, env_creator, name):
        self.agent_creator = agent_creator
        self.env_creator = env_creator
        self.folder_name = "evaluate_{}".format(name)
        if os.path.exists(self.folder_name):
            shutil.rmtree(self.folder_name)
        os.mkdir(self.folder_name)
        self.weight_file = "{}/weights.h5f".format(self.folder_name)

    def train(self, repeat=100, showDiagram=True):
        best_total_reward = float('-inf')
        for i in range(repeat):
            agent = self.agent_creator()
            print(agent.metrics_names)
            env = self.env_creator()
            filename = os.path.join(self.folder_name, "train_{}_reward.csv".format(i+1))
            steps = env.frame_bound[1] - env.frame_bound[0] -1
            print("starting train {}".format(i+1))
            
            agent.fit(env, nb_steps=steps, visualize=False, verbose=0, callbacks=[CumulativeRewardLogger(env, filename, interval=steps/5)])
            data = np.loadtxt(filename)
            max_data = np.max(data, axis=0)
            min_data = np.min(data, axis=0)
            mean_data = np.mean(data, axis=0)
            print()
            print("train {} completed. total_reward: {} total_profit: {}".format(i+1, env._total_reward, env._total_profit))
            print("min reward: {}, max reward: {}, mean_reward: {}".format(min_data[1], max_data[1], mean_data[1]))
            if showDiagram:
                plt.cla()
                env.render_all()
                plt.show()
                plt.plot(data[:, [0]], data[:, [2]])
                plt.xlabel('steps')
                plt.ylabel('Cummulative Reward')
                plt.show()
            print()
            if env._total_reward > best_total_reward:
                agent.save_weights(self.weight_file, overwrite=True)


    def process_train_result(self, showDiagram=True):
        files = glob.glob("{}/train_*_reward.csv".format(self.folder_name))
        all_data = []
        for f in files:
            all_data.append(np.loadtxt(f))
        stacks = np.stack(all_data)
        mean_reward = np.mean(stacks, axis=0)
        steps = mean_reward[:, [0]]
        average_cummulative_reward = mean_reward[:, [2]]

        plt.clf()
        plt.close()
        plt.style.use('ggplot')
        plt.plot(steps, average_cummulative_reward)
        plt.xlabel('steps')
        plt.ylabel('Average Cummulative Reward')
        plt.title('Average Cummulative Reward accross steps')
        plt.savefig("{}/acr.png".format(self.folder_name))
        if showDiagram:
            plt.show()
    
    def test(self, env, showDiagram=True):
        agent = self.agent_creator()
        agent.load_weights(self.weight_file)
        steps = env.frame_bound[1] - env.frame_bound[0] -1
        filename = os.path.join(self.folder_name, "test_reward.csv")
        agent.test(env, visualize=False, callbacks=[TestCallback(env, filename, interval=steps/5)])

        data = np.loadtxt(filename)
        max_data = np.max(data, axis=0)
        min_data = np.min(data, axis=0)
        mean_data = np.mean(data, axis=0)
        print()
        print("test completed. total_reward: {} total_profit: {}".format(env._total_reward, env._total_profit))
        print("min reward: {}, max reward: {}, mean_reward: {}".format(min_data[1], max_data[1], mean_data[1]))
        print()
        steps = data[:, [0]]
        average_cummulative_reward = data[:, [2]]
        plt.clf()
        plt.close()

        plt.style.use('ggplot')
        plt.plot(steps, average_cummulative_reward)
        plt.xlabel('steps')
        plt.ylabel('Cummulative Reward')
        plt.title('Cummulative Reward accross steps')
        plt.savefig("{}/test_acr.png".format(self.folder_name))
        if showDiagram:
            plt.show()


            


