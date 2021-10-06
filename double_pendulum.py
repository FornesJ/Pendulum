from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation


class DoublePendulum:
    def __init__(self, L1=1, L2=1, M1=1, M2=1, g=9.81):
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2
        self.g = g

    def __call__(self, t, y):
        theta1, omega1, theta2, omega2 = y
        dtheta = theta2 - theta1
        dtheta1_dt = omega1
        dtheta2_dt = omega2

        domega1_dt = (self.L1*omega1**2*np.sin(dtheta)*np.cos(dtheta)
        +self.g*np.sin(theta2)*np.cos(dtheta)
        +self.L2*omega2**2*np.sin(dtheta)
        -2*self.g*np.sin(theta1))/(2*self.L1 - self.L1*np.cos(dtheta)**2)

        domega2_dt = (-self.L2*omega2**2*np.sin(dtheta)*np.cos(dtheta)
        +2*self.g*np.sin(theta1)*np.cos(dtheta)
        -2*self.L1*omega1**2*np.sin(dtheta)
        -2*self.g*np.sin(theta2))/(2*self.L2 - self.L2*np.cos(dtheta)**2)

        return dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt

    def solve(self, y0, T, dt, angle='rad'):
        self.dt = dt
        self.T = T
        rad1 = y0[0]
        rad2 = y0[2]
        if angle == 'deg':
            rad1 = np.radians(rad1)
            rad2 = np.radians(rad2)
        sol = solve_ivp(self, [0, self.T], 
        (rad1, y0[1], rad2, y0[3]), method="Radau", 
        t_eval = np.linspace(0,self.T,round(self.T/self.dt)), 
        max_step = self.dt)

        self._theta1 = sol.y[0]
        self._omega1 = sol.y[1]
        self._theta2 = sol.y[2]
        self._omega2 = sol.y[3]
        self._t = sol.t

    @property
    def t(self):
        if self._t[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._t

    @property
    def theta1(self):
        if self._theta1[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._theta1

    @property
    def theta2(self):
        if self._theta2[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._theta2

    # omega1 and omega2 will not be used, but we will still store them as properties

    @property
    def omega1(self):
        if self._omega1[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._omega1

    @property
    def omega2(self):
        if self._omega2[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._omega2

    @property
    def x1(self):
        x1 = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            x1[i] = self.L1*np.sin(self.theta1[i])
        return x1
    
    @property
    def y1(self):
        y1 = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            y1[i] = -self.L1*np.cos(self.theta1[i])
        return y1

    @property
    def x2(self):
        x2 = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            x2[i] = self.x1[i] + self.L2*np.sin(self.theta2[i])
        return x2
    
    @property
    def y2(self):
        y2 = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            y2[i] = self.y1[i] - self.L2*np.cos(self.theta2[i])
        return y2

    @property
    def potential(self):
        P = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            P1 = self.M1*self.g*(self.y1[i] + self.L1)
            P2 = self.M2*self.g*(self.y2[i] + self.L1 + self.L2)
            P[i] = P1 + P2
        return P
    
    @property
    def vx1(self):
        vx1 = np.gradient(self.x1, self.dt)
        return vx1

    @property
    def vx2(self):
        vx2 = np.gradient(self.x2, self.dt)
        return vx2

    @property
    def vy1(self):
        vy1 = np.gradient(self.y1, self.dt)
        return vy1

    @property
    def vy2(self):
        vy2 = np.gradient(self.y2, self.dt)
        return vy2

    @property
    def kinetic(self):
        K = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            K1 = (1/2)*self.M1*(self.vx1[i]**2+self.vy1[i]**2)
            K2 = (1/2)*self.M2*(self.vx2[i]**2+self.vy2[i]**2)
            K[i] = K1 + K2
        return K

    @property
    def totalE(self):
        E = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            E[i] = self.potential[i] + self.kinetic[i]
        return E

    def create_animation(self):
        # Create empty figure
        fig = plt.figure()
            
        # Configure figure
        plt.axis('equal')
        plt.axis('off')
        plt.axis((-3, 3, -3, 3))
            
        # Make an "empty" plot object to be updated throughout the animation
        self.pendulums, = plt.plot([], [], 'o-', lw=2)
            
        # Call FuncAnimation
        self.animation = FuncAnimation(fig,
                            self._next_frame,
                            frames=range(len(self.x1)), 
                            repeat=None,
                            interval=10, 
                            blit=True)

    def _next_frame(self, i):
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.y1[i], self.y2[i]))
        return self.pendulums,

    def show_animation(self):
        plt.show()

    def save_animation(self, file_name):
        self.animation.save(file_name, fps=60)





if __name__ == "__main__":
    instance = DoublePendulum()
    instance.solve(y0=(np.pi/3,0.15,np.pi/6,0.10), T=10, dt=0.1)
    instance.create_animation()
    instance.show_animation()
    #instance.save_animation(file_name = "pendulum_motion.mp4")

    plt.plot(instance.t,instance.kinetic, label='kinetic')
    plt.plot(instance.t,instance.potential, label='potential')
    plt.plot(instance.t,instance.totalE, label='total energy')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.show()
    



    


    

    