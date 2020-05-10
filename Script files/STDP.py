import numpy as np

class STDP:
    def __init__(self, t_back, t_front):
        self.t_backward = t_back
        self.t_forward = t_front
        
        self.A_plus = 0.5
        self.A_minus = 0.5
        self.tau_plus = 5
        self.tau_minus = 5
        self.scale = 1
        self.lr = 0.1
        self.w_max = 2*self.scale
        self.w_min = -2*self.scale
        
    def get_del_w(self, t): #del_t = t_pre - t_post

	    if t>0:
		    return -self.A_minus*np.exp(-float(t)/self.tau_minus)
	    if t<=0:
		    return self.A_plus*np.exp(float(t)/self.tau_plus)

    def update_w(self, w, del_t): 

        del_w = self.get_del_w(del_t)

        if del_w < 0:
            #e = abs(self.w_min)-w if w<0 else abs(self.w_min)+w #difference 
            return w + self.lr*del_w*(w-self.w_min)*self.scale

        elif del_w > 0:
            return w + self.lr*del_w*(self.w_max-w)*self.scale
            
