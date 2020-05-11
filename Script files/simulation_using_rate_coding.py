import numpy as np
import matplotlib.pyplot as plt

from neuron_models import LIF
from spike_encoding import spike_encoding
from STDP import STDP

signal_timesteps = 20
input_dimension = 3
original_signal = np.random.randint(0,100, (input_dimension, signal_timesteps))

#Encoding
encoder = spike_encoding(scheme='rate_coding', input_range=(0,100))
signal = encoder.encode(original_signal)

#Network
dt = 0.125 #ms
m, n = input_dimension, 2

h_layer = []
for i in range(n):
    neuron = LIF(threshold=0.3, dt=dt, Cm=(i+1)*20)
    h_layer.append(neuron)
    
synapse = np.random.normal(loc=1,scale=0.1,size=(n,m))

#Simulation

epochs = 200
T = signal.shape[1] #Total time-steps in encoded signal

stdp = STDP(-10,10)
stdp.w_min = 0
stdp.lr = 0.01

fig, (ax_s, ax_a) = plt.subplots(2,1)

for e in range(epochs):
    
    activations=[]
    for i in range(n):
        activations.append([0])
        h_layer[i].initialize()

    for t in range(T):
        for idx,neuron in enumerate(h_layer):
            input_I = np.dot(synapse[idx],signal[:,t])
            activations[idx].append(neuron.update(input_I, t))

        for idx,neuron in enumerate(h_layer):
            if neuron.Vm == neuron.V_spike:
                for i in range(m):
                    for t1 in range(-1,stdp.t_backward,-1):
                        if 0<=t+t1<T:
                            if signal[i][t+t1] == 1:
                                synapse[idx][i] = stdp.update_w(synapse[idx][i], t1)


                    for t1 in range(1,stdp.t_forward,1):
                        if 0<=t+t1<T:
                            if signal[i][t+t1] == 1:
                                synapse[idx][i] = stdp.update_w(synapse[idx][i], t1)

    plt.pause(0.01)
    ax_s.cla()
    ax_a.cla()
    for p in range(n):
        ax_s.plot(synapse[p])        
        ax_a.plot(activations[p])    
    print(e)

plt.show()
print('done')
        
