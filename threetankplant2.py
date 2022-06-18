# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:55:29 2022

@author: lucky
"""
import numpy as np
import matplotlib.pyplot as plt
import gym

class ThreeTank(gym.Env):
    
    def __init__(self,state_space):
        self.action_space = 1
        self.observation_space = state_space
        self.time = np.linspace(-5,20, 51)
        self.prev_e = 0
    def plantmodel(self,t,x,u):
        
        xdot = np.zeros(len(x))
        xdot[0] = -x[0] + u 
        xdot[1] = x[0] - x[1]
        xdot[2] = x[1] - x[2]
        
        return xdot
        
        
        
    def step(self,u,t,x0,r):
        
        
        
        Sol_t, Sol_y = RK4_solver(self.plantmodel,t ,x0, 100,u)
        x = Sol_y[-1]
        y = Sol_y[-1,-1]
        
        if abs(r-y) < 0.01:
            done = True
        
        else:
            done = False
        

            
        #d_error = self.prev_e + error*0.5
        reward = - abs(r-y) #-1*(1+error**2)
        #self.prev_e = error
        return x,y,reward,done
            
        
    def reset(self):
        
        return np.zeros(self.observation_space)


def RK4_solver(dydt,tspan,y0,n,u):

    if ( np.ndim ( y0 ) == 0 ):
      m = 1
    else:
      m = len ( y0 )

    tfirst = tspan[0]
    tlast = tspan[1]
    dt = ( tlast - tfirst ) / n
    t = np.zeros ( n + 1 )
    y = np.zeros ( [ n + 1, m ] )
    t[0] = tspan[0]
    y[0,:] = y0

    for i in range ( 0, n ):

      f1 = dydt ( t[i],            y[i,:] ,u)
      f2 = dydt ( t[i] + dt / 2.0, (y[i,:] + dt * (f1 / 2.0)) ,u)
      f3 = dydt ( t[i] + dt / 2.0, (y[i,:] + dt * (f2 / 2.0)),u )
      f4 = dydt ( t[i] + dt,       y[i,:] + dt * f3 ,u)
      
      
      t[i+1] = t[i] + dt
     
      y[i+1,:] = y[i,:] + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

    return t, y

"""
def plantmodel(t,x,u):
    
    xdot = np.zeros(len(x))
    xdot[0] = -x[0] + u 
    xdot[1] = x[0] - x[1]
    xdot[2] = x[1] - x[2]
    
    return xdot


def IMC_PID(uk2,rk3,rk2,rk1):
    
    ti = 3 #minutes
    td = 0.4 #minutes
    kc = 1#5.379 #Adjusted Kc (minutes)
    deltaT = 0.5 # sample time in minutes
    
    b1 = -kc*(1+(2*td/deltaT))
    b2 = kc*(1+(deltaT/ti) + (td/deltaT))
    b0 = kc*td/deltaT
    uk = uk2 + b2*rk3 + b1*rk2 + b0*rk1

    return uk



x0 = np.zeros(3)
time = np.linspace(-5,20, 51)


y = np.zeros((len(time)+3,1))
r = np.zeros((len(time)+3,1))
error = np.zeros((len(time)+3,1))
u = np.zeros((len(time)+3,1))

dummy = 0

for k in range(len(time)):
    
    if time[k] < 1:
        r[k+3] = 0
    else:
        r[k+3] = 1
        
        
    error[k+3] = r[k+3] - y[k+2]
    
    u[k+3] = IMC_PID(u[k+2], error[k+3], error[k+2], error[k+1])
    t = [time[k], time[k]+0.5]
    Sol_t, Sol_y = RK4_solver(plantmodel,t ,x0, 40,u[k+3])
    x0 = Sol_y[-1]
    y[k+3] = Sol_y[-1,-1]

y = y[3:]
r = r[3:]
u = u[3:]
#%%

fig, (ax1,ax2) = plt.subplots(2,1,dpi=500)

ax1.plot(time,y,linewidth=0.8,label="Output")
ax1.step(time,r,"--", linewidth=0.8,label="Setpoint")
ax1.legend()
ax1.set_ylabel("Height of Tank 3")

ax2.step(time,u,"--",linewidth=0.8)
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Manipulated Input")
fig.suptitle("Discrete Response with Kc=1")
plt.tight_layout()
plt.show()

"""