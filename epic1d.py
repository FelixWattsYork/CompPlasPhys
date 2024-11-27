#!/usr/bin/env python3
#
# Electrostatic PIC code in a 1D cyclic domain

from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max
import numpy as np
import pickle
import scipy 
import math
from scipy.optimize import curve_fit
import time

import matplotlib.pyplot as plt # Matplotlib plotting library

try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

# Need an FFT routine, either from SciPy or NumPy
try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    k1 = f(y0, *args)
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)

    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

def calc_density(position, ncells, L):
    """ Calculate charge density given particle positions
    
    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    # This is a crude method and could be made more efficient
    
    density = zeros([ncells])
    nparticles = len(position)
    
    dx = L / ncells       # Uniform cell spacing
    for p in position / dx:    # Loop over all the particles, converting position into a cell number
        plower = int(p)        # Cell to the left (rounding down)
        offset = p - plower    # Offset from the left
        density[plower] += 1. - offset
        density[(plower + 1) % ncells] += offset
    # nparticles now distributed amongst ncells
    density *= float(ncells) / float(nparticles)  # Make average density equal to 1
    return density

def periodic_interp(y, x):
    """
    Linear interpolation of a periodic array y at index x
    
    Input

    y - Array of values to be interpolated
    x - Index where result required. Can be an array of values
    
    Output
    
    y[x] with non-integer x
    """
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    xl = floor(x).astype(int) # Left index
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx

def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n
    
    # Modify frequencies by dividing by ik
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f).real # Reverse Fourier Transform
   

def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f) // 2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)
    
    # Subtract ion density to get total charge density
    rho = density - 1.
    
    # Calculate electric field
    E = -fft_integrate(rho)*dx
    
    # Interpolate E field at particle locations
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################

def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):
    
    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)
    
    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        while stepping:
            # Maximum distance a particle can move is one cell
            dt = cfl * dx / max(abs(vel))
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt
            
        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]
        
        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)
        
    return pos, vel

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")
            
            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]
            
            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]
            
            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]
            
            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        self.phase_plot.set_data(pos, vel) # Update the plot
        self.density_plot.set_data(linspace(0, L, ncells), d)
        self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)
        

class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []
        
    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        print(f"Time: {t} First: {fh}")
        
        self.t.append(t)
        self.firstharmonic.append(fh)

####################################################################
# 
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping
    
    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )
        
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other
    
    return pos,vel

####################################################################

Load  = 1
Save_name = "run.pickle"
Load_name = "good.pickle"

import pickle
 
class MyClass():
    def __init__(self,pos,vel,npart,ncells,cal_time,L,s):
        self.pos = pos
        self.vel = vel
        self.npart = npart
        self.ncells = ncells
        self.cal_time = cal_time
        self.L = L
        self.s = s
 
def save_object(obj,filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

    
def  damping(t,a,d):
    return a*np.exp(-d*t)

n = 5  # Number of pairs (10^n, 5*10^n) you want
powers_of_ten = 10 ** np.arange(n)
array = np.empty(2 * n, dtype=int)  # Create an empty array of the required size
array[0::2] = powers_of_ten        # Assign 10^n to even indices
array[1::2] = 5 * powers_of_ten    # Assign 5 * 10^n to odd indices

particle_numbers = array
cell_numbers = array

def run_list():
    for npart in particle_numbers:
        for ncells in cell_numbers:
            L = 4.*pi
            s = Summary()                 # Calculates, stores and prints summary info

            diagnostics_to_run = [s]   # Remove p to get much faster code!

            # Run the simulation
            pos, vel = run(pos, vel, L, ncells, 
                        out = diagnostics_to_run,        # These are called each output step
                        output_times=linspace(0.,20,50)) # The times to output
            obj = MyClass(pos,vel,npart,ncells,cal_time,L,s)
            Save_name = "pn{}_cn{}".format(npart,ncells)
            save_object(obj,"data/"+Save_name)

if __name__ == "__main__":
    if Load == 0:
        # Generate initial condition
        # 
        npart = 10000   
        if False:
            # 2-stream instability
            L = 100
            ncells = 20
            pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
        else:
            # Landau damping
            L = 4.*pi
            ncells = 20
            pos, vel = landau(npart, L)
        # Create some output classes
        p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
        s = Summary()                 # Calculates, stores and prints summary info

        diagnostics_to_run = [p, s]   # Remove p to get much faster code!
        

        # Run the simulation
        pos, vel = run(pos, vel, L, ncells, 
                    out = diagnostics_to_run,        # These are called each output step
                    output_times=linspace(0.,20,50)) # The times to output
        obj = MyClass(pos,vel,ncells,L,s)
        save_object(obj,Save_name)
    elif Load == 1:
        obj = load_object(Load_name)
        pos,vel,ncells,L,s = obj.pos,obj.vel,obj.ncells,obj.L,obj.s
        p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    # Summary stores an array of the first-harmonic amplitude
    # Make a semilog plot to see exponential damping
    extrema = scipy.signal.argrelextrema(np.array(s.firstharmonic),np.greater)[0]
    extrema_t = [s.t[0]]
    extrema_harm = [s.firstharmonic[0]]
    first_largest = -1
    sign = -1
    for n in extrema:
        extrema_t.append(s.t[n])
        extrema_harm.append(s.firstharmonic[n])
    for n in range (1,len(extrema_harm)):
            if extrema_harm[n]>extrema_harm[n-1]:
                first_largest = extrema[n]
                sign = n
                break
    print(sign)
    extrema_harm = np.array(extrema_harm)
    extrema_t = np.array(extrema_t)
    sig  = np.array(s.firstharmonic[0:first_largest])
    sig_time = np.array(s.t[0:first_largest])
    noise = np.array(s.firstharmonic[first_largest:-1])
    noise_time = np.array(s.t[first_largest:-1])
    sig_peaks = extrema_harm[0:sign]
    sig_t = extrema_t[0:sign]

    #calculating spacing between peaks
    ave = []
    for n in range (1,len(extrema_t)):
        ave.append(extrema_t[n]-extrema_t[n-1])
    average = np.mean(ave)
    error = np.std(ave)
    popt,pcov = curve_fit(damping,sig_t,sig_peaks)
    print("pot ",*popt)
    print("average ", average)
    print("frequency ",1/average)
    plt.figure()
    plt.plot(sig_time,sig)
    plt.plot(noise_time,noise)
    plt.scatter(sig_t, sig_peaks)
    plt.plot(sig_t,damping(sig_t,*popt))
    plt.xlabel("Time [Normalised]")
    plt.ylabel("First harmonic amplitude [Normalised]")
    plt.yscale('log')
    plt.ioff() # This so that the windows stay open
    plt.show()