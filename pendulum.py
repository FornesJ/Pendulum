from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt 

# Creating a class Pendulum
class Pendulum:
    def __init__(self, L=1, M=1, g=9.81): # Length, mass and gravity
        self.L = L
        self.M = M
        self.g = g

    # Defining a solve method using solving_ivp
    def solve(self, y0, T, dt, angle='rad'):
        self.dt = dt
        self.T = T
        rad = y0[0]
        if angle == 'deg':
            rad = np.radians(rad)
        sol = solve_ivp(self, [0, self.T], 
        (rad, y0[1]), t_eval = np.linspace(0,self.T,round(self.T/self.dt)), max_step = self.dt)
        self._theta = sol.y[0]
        self._omega = sol.y[1]
        self._t = sol.t

    # Properties for t, theta and omega
    @property
    def t(self):
        if self._t[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._t

    @property
    def theta(self):
        if self._theta[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._theta

    @property
    def omega(self):
        if self._omega[0] == None:
            raise ValueError ('Did you call solve?')
        else:
            return self._omega

    # Function call
    def __call__(self,t,y):
        theta, omega = y
        dw = (-self.g/self.L)*np.sin(theta)
        return omega, dw

    # Properties for x, y, potential, vy, vx, kinetic and total energy
    @property
    def x(self):
        x = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            x[i] = self.L*np.sin(self.theta[i])
        return x
    
    @property
    def y(self):
        y = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            y[i] = -self.L*np.cos(self.theta[i])
        return y

    @property
    def potential(self):
        P = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            P[i] = self.M*self.g*(self.y[i] + self.L)
        return P

    @property
    def vx(self):
        vx = np.gradient(self.x, self.dt)
        return vx

    @property
    def vy(self):
        vy = np.gradient(self.y, self.dt)
        return vy

    @property
    def kinetic(self):
        K = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            K[i] = (1/2)*self.M*(self.vx[i]**2+self.vy[i]**2)
        return K

    @property
    def totalE(self):
        E = np.zeros(len(self.t))
        for i in range(0,len(self.t)):
            E[i] = self.potential[i] + self.kinetic[i]
        return E

# Creating subclass of Pendulum, DampenedPendulum
class DampenedPendulum(Pendulum):
    # Rewriting the consctructor from Pendulum, to add a variable B
    def __init__(self, B, L=1, M=1, g=9.81):
        super().__init__(L=1, M=1, g=9.81)
        self.B = B

    # Overwriting the old __call__ method
    def __call__(self, t, y):
        theta, omega = y
        dw = -(self.g/self.L)*np.sin(theta) - (self.B/self.M)*omega
        return omega, dw

# Testing our code with an example
if __name__ == "__main__":
    instance = Pendulum()
    instance.solve((np.pi/3,0.15), 5, 0.05)
    fig, ax = plt.subplots(nrows=2, figsize=(6,8))
    ax[0].plot(instance.t, instance.theta, label='Motion over time')
    ax[0].legend()
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('theta')
    ax[1].plot(instance.t,instance.kinetic, label='kinetic')
    ax[1].plot(instance.t,instance.potential, label='potential')
    ax[1].plot(instance.t,instance.totalE, label='total energy')
    ax[1].legend()
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('energy')
    plt.show()

    instance2 = DampenedPendulum(2)
    instance2.solve((np.pi/3,0.15), 5, 0.05)
    fig, ax = plt.subplots(nrows=2, figsize=(6,8))
    ax[0].plot(instance2.t, instance2.theta, label='Motion over time')
    ax[0].legend()
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('theta')
    ax[1].plot(instance2.t,instance2.kinetic, label='kinetic')
    ax[1].plot(instance2.t,instance2.potential, label='potential')
    ax[1].plot(instance2.t,instance2.totalE, label='total energy')
    ax[1].legend()
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('energy')
    plt.show()