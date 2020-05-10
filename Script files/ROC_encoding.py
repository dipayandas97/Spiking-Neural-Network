import numpy as np

class ROC:
    
    def __init__(self):
        self.time_window = 10
        
    def encode(self, signal):
    
        if len(signal.shape)!=2:
            raise Exception('Input signal should have more than one input dimension!')

        spike_train = np.zeros((signal.shape[1], signal.shape[0], self.time_window+1))

        for t in range(signal.shape[1]):
            s = signal[:,t]
            s = np.max(s) - s
            latency = self.time_window * ((s - np.min(s))/(np.max(s) - np.min(s)))

            for i in range(latency.shape[0]): #iterate over each dimension of data
                spike_train[t][i][int(latency[i])] = 1 

        #Total encoded data

        seq = spike_train[0]

        for w in spike_train[1:]:
            seq = np.hstack((seq, w))

        return seq
