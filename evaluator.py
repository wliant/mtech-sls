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
    def __init__(self, env, filename, metricFile, interval=20):
        TrainIntervalLogger.__init__(self, interval)
        self.cumulative_reward = 0
        self.filename = filename
        self.metricFile = metricFile
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
        np_metric_data = np.array(self.metrics)
        np.savetxt(self.metricFile, np_metric_data)

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
    def __init__(self, agent_creator, env_creator, name, quiet=False):
        self.agent_creator = agent_creator
        self.env_creator = env_creator
        self.folder_name = "evaluate_{}".format(name)
        if os.path.exists(self.folder_name):
            shutil.rmtree(self.folder_name)
        os.mkdir(self.folder_name)
        self.weight_file = "{}/weights.h5f".format(self.folder_name)
        self.log_file = os.path.join(self.folder_name, "log.log")
        self.quiet = quiet


    def log(self, text):
        with open(self.log_file, 'a') as outfile:
            outfile.write(text + '\n')
        if not self.quiet:
            print(text)

    def train(self, repeat=100):
        train_start = timeit.default_timer()

        best_total_reward = float('-inf')
        for i in range(repeat):
            agent = self.agent_creator()
            #print(agent.metrics_names)
            env = self.env_creator()
            rewardFilename = os.path.join(self.folder_name, "train_{}_reward.csv".format(i+1))
            rewardFigureFile = os.path.join(self.folder_name, "train_{}_reward.png".format(i+1))
            metricFile = os.path.join(self.folder_name, "train_{}_metric.csv".format(i+1))
            renderAllFile = os.path.join(self.folder_name, "train_{}_render_all.png".format(i+1))
            
            steps = env.frame_bound[1] - env.frame_bound[0] -1
            self.log("starting train {}".format(i+1))
            
            agent.fit(env, nb_steps=steps, visualize=False, verbose=0, callbacks=[CumulativeRewardLogger(env, rewardFilename, metricFile, interval=steps/5)])

            data = np.loadtxt(rewardFilename)
            max_data = np.max(data, axis=0)
            min_data = np.min(data, axis=0)
            mean_data = np.mean(data, axis=0)
            duration = timeit.default_timer() - train_start
            self.log("train {} completed. took {:.3f} seconds, total_reward: {} total_profit: {}".format(i+1, duration, env._total_reward, env._total_profit))
            self.log("min reward: {}, max reward: {}, mean_reward: {}".format(min_data[1], max_data[1], mean_data[1]))

            metrics = np.loadtxt("evaluate_dqn/train_1_metric.csv")
            col = np.array([[i+1] for i in range(len(metrics))])
            m = np.zeros((metrics.shape[0],metrics.shape[1]+1))
            m[:, 1:] = metrics
            m[:, [0]] = col
            metrics = m[~np.isnan(m).any(axis=1)]
            plt.close()
            env.render_all()
            plt.savefig(renderAllFile)
            plt.show()
            
            plt.close()
            plt.plot(data[:, [0]], data[:, [2]])
            plt.xlabel('steps')
            plt.ylabel('Cummulative Reward')
            plt.savefig(rewardFigureFile)
            plt.show()
            
            plt.close()

            j = 1
            for metric_name in agent.metrics_names:
                f = os.path.join(self.folder_name, "train_{}_{}.png".format(i+1, metric_name))
                plt.close()
                plt.plot(metrics[:, [0]], metrics[:, [j]])
                j+=1
                plt.xlabel('steps')
                plt.ylabel(metric_name)
                plt.savefig(f)
                plt.show()
                

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
        plt.title('Average Cummulative Reward across experiment')
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
        plt.close()
        plt.cla()
        env.render_all()
        plt.show()
        plt.style.use('ggplot')
        plt.plot(steps, average_cummulative_reward)
        plt.xlabel('steps')
        plt.ylabel('Cummulative Reward')
        plt.title('Cummulative Reward accross steps')
        plt.savefig("{}/test_acr.png".format(self.folder_name))
        if showDiagram:
            plt.show()


            


