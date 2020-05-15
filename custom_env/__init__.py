
from gym_anytrading.envs import ForexEnv
import numpy as np
class ForexV0FullFeature(ForexEnv):
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'Close'].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['Open','High','Low','Close','Volume']].to_numpy()[start:end]
        
        diffs = np.diff(signal_features, axis=0)
        diffs_2nd = np.diff(diffs, axis=0)

        diffs = np.insert(diffs, 0, np.zeros(diffs.shape[1]), axis=0)
        diffs_2nd = np.insert(diffs_2nd, 0, np.zeros(diffs_2nd.shape[1]), axis=0)
        diffs_2nd = np.insert(diffs_2nd, 0, np.zeros(diffs_2nd.shape[1]), axis=0)
        
        signal_features = np.column_stack((signal_features, diffs, diffs_2nd))

        return prices, signal_features

