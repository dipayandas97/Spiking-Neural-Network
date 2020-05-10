
import numpy as np
import matplotlib.pyplot as plt
import time

signal = np.random.normal(loc=0,scale=2,size=100)
start = time.time()

while True:
    plt.ylim(-7,7)
    plt.plot(signal)
    plt.pause(1e-10)
    plt.cla()
    
    signal[:-1] = signal[1:]
    signal[-1] = np.random.normal(loc=0, scale=2, size=1)[0]
    
    print(time.time()-start)
    start=time.time()

