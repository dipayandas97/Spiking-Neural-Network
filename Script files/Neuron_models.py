class LIF:
    def __init__(self, 
                threshold=1, 
                dt=0.125,
                Vres=0,
                tau_ref=4,
                Rm=1,
                Cm=10):
        #simulation parameters
        self.dt = dt

        #LIF parameters
        self.Vres = Vres                     #resting potential (V)
        self.Vm = self.Vres                  #current potential (V)
        self.t_rest = -1                     #initial resting time point
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = threshold                 #(V)

        self.Rm = Rm                          #kOhm
        self.Cm = Cm                          #uF
        self.tau_m = self.Rm * self.Cm                 #(ms)
        self.V_spike = threshold+2                   #spike delta (V)
            
    def update(self, I, time_stamp):
        if time_stamp > self.t_rest:
            self.Vm = self.Vm + (((I*self.Rm - self.Vm) / self.tau_m) * self.dt)
            
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.t_rest = time_stamp + self.tau_ref
        else:
            self.Vm = self.Vres
        return self.Vm
    
    def initialize(self):
        self.Vm = self.Vres
        self.t_rest = -1
