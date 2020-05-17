#Creating an object of this class, returns a network instance with the specified parameters
import numpy as np
from neuron_models import LIF

class SNN_Model:
    def __init__(self, input_dim, neurons_per_layer, neuron_type='LIF', threshold=1, dt=0.125, tau_ref=4, Rm=1, Cm=10):
        self.input_dim = input_dim                              #EX: 4
        self.neurons_per_layer = neurons_per_layer              #EX: (6,2)
        self.num_layers = len(self.neurons_per_layer)        
        
        self.neuron_type = 'LIF'
        self.threshold=threshold
        self.dt=dt
        self.tau_ref=tau_ref,
        self.Rm=Rm
        self.Cm=Cm
        
        self.neuron_list = self.get_neuron_list()
        self.synapse_list = self.get_synapse_list()

    def get_neuron_list(self):
        neuron_list = []
        for l in self.neurons_per_layer:
            n_l = []
            for i in range(l):
                neuron = LIF(threshold=self.threshold, dt=self.dt, tau_ref=self.tau_ref, Rm=self.Rm, Cm=self.Cm)
                n_l.append(neuron)
            neuron_list.append(n_l)
        return neuron_list
                        
    def get_synapse_list(self):
        synapse_list = []
        for l in range(self.num_layers):
            if l==0:
                synapse = np.random.uniform(-1,1,size=(self.neurons_per_layer[l], self.input_dim))
            else:
                synapse = np.random.uniform(-1,1,size=(self.neurons_per_layer[l], self.neurons_per_layer[l-1]))
            synapse_list.append(synapse)
        return synapse_list
    
    def initialize_activation_list(self, sim_time):
        activation_list = []
        for l in range(self.num_layers+1):
            if l==0:
                act = np.zeros((self.input_dim, sim_time))
            else:
                act = np.zeros((self.neurons_per_layer[l-1], sim_time))
            activation_list.append(act)
        return activation_list    
                                
    def predict(self, input_signal):                              #signal is a 2D matrix shaped: (feature,timesteps)
        simulation_time = input_signal.shape[1]
        activations_list = self.initialize_activation_list(simulation_time)   #list of 2D matrices
        activations_list[0][:,:] = input_signal[:,:]                       
     
        #re-initialize all neurons in the model
        for l in range(self.num_layers):
            for neuron in self.neuron_list[l]:
                neuron.initialize()        

        #Simulate
        for t in range(simulation_time):
            for l in range(self.num_layers):
                synapse = self.synapse_list[l]
                signal = activations_list[l][:,t]
                
                for n_id,neuron in enumerate(self.neuron_list[l]):
                    input_val = np.dot(synapse[n_id],signal)
                    activations_list[l+1][n_id,t] = neuron.update(input_val, t)
                    
        return activations_list #returns all layer activations : [input_layer:output_layer]
                
    def set_synapses(self, synapse_list): #list of synapse matrices
        for l in range(self.num_layers):
            if self.synapse_list[l].shape == synapse_list[l].shape:
                self.synapse_list[l][:,:] = synapse_list[l][:,:]
            else:
                raise Exception('Synapse dimension not matched. For layer ',l,'expected dimension is',self.synapse_list[l].shape,'but received dimension',synapse_list[l].shape)
            
                            
