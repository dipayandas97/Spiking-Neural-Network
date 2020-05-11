import numpy as np

class spike_encoding:

    def __init__(self, scheme='rate_coding', time_window=100, input_range=(0,255), output_freq_range=(10,200)):
        
        self.scheme = scheme
        
        if scheme is 'rate_coding':
            self.min_input, self.max_input = input_range[0], input_range[1]
            self.min_output, self.max_output = output_freq_range[0], output_freq_range[1]
            self.time_window = time_window
            
        elif scheme is 'rank_order_coding':
            self.time_window = time_window
        else:
            raise Exception('No valid spike encoding scheme selected!')
            
        
    def encode(self, signal):
    
        if self.scheme is 'rate_coding':
            if len(signal.shape)>1:
                repeat = signal.shape[0] 
            else:
                raise Exception('encoder() excepts signal with 2D. Reshape to (1,-1) if signal is 1D')
                
            total_spikes = []
            for r in range(repeat):
                spike_train = []        
                for s in signal[r]:
                    freq = ((s-self.min_input)/(self.max_input-self.min_input)) * (self.max_output-self.min_output) + self.min_output 
                    t = (1 / freq) * 1000 #ms
                    
                    spikes = np.zeros(self.time_window)
                    k=0
                    while k<self.time_window:
                        spikes[k] = 1
                        k += int(t)
                    spike_train.append(spikes)
                spike_train = np.hstack(([x for x in spike_train]))  
                total_spikes.append(spike_train) 

            return np.asarray(total_spikes)
        
        elif self.scheme is 'rank_order_coding':
        
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
 
