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

def  damping(t,a,d):
    return a*np.exp(-d*t)

def growth(t,a,b,c,d):
    ones_array = np.ones(len(t))
    return a*np.tanh(b*(t+c*ones_array))+d

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
Save_name = "data_TwoStream/pn50000_cn20.pickle"
Load_name = "data_TwoStream/pn50000_cn20.pickle"
Run_type = "TwoStream"

import pickle
 
class Run_Outcome():
    def __init__(self,pos,vel,npart,ncells,cal_time,L,s,noise_level=0,frequency=0,frequency_error=0,damping_rate=0,damping_rate_error=0,run_type = "Unknown"):
        self.pos = pos
        self.vel = vel
        self.npart = npart
        self.ncells = ncells
        self.cal_time = cal_time
        self.L = L
        self.s = s
        self.noise_level = noise_level
        self.frequency = frequency
        self.frequency_error = frequency_error
        self.damping_rate = damping_rate
        self.damping_error = damping_rate_error
        self.run_type = run_type
        self.growth_rate = 0
        self.growth_point = 0


    def calculate_values(self):
        if self.run_type == "Landau":
            extrema = scipy.signal.argrelextrema(np.array(self.s.firstharmonic),np.greater)[0]
            extrema_t = [self.s.t[0]]
            extrema_harm = [self.s.firstharmonic[0]]
            first_largest = -1
            sign = -1
            for n in extrema:
                extrema_t.append(self.s.t[n])
                extrema_harm.append(self.s.firstharmonic[n])
            for n in range (1,len(extrema_harm)-1):
                    if extrema_harm[n]>extrema_harm[n-1]:
                        first_largest = extrema[n]
                        sign = n
                        break
            extrema_harm = np.array(extrema_harm)
            extrema_t = np.array(extrema_t)
            sig  = np.array(self.s.firstharmonic[0:first_largest])
            sig_time = np.array(self.s.t[0:first_largest])
            noise = np.array(self.s.firstharmonic[first_largest:-1])
            noise_time = np.array(self.s.t[first_largest:-1])
            noise_average_sqaure = np.mean(noise**2)
            sig_average_sqaure = np.mean(sig**2)


            if noise_average_sqaure>0:
                self.noise_level = sig_average_sqaure/noise_average_sqaure

            sig_peaks = extrema_harm[0:sign]
            sig_t = extrema_t[0:sign]

            #calculating spacing between peaks
            ave = []
            for n in range (1,len(extrema_t)):
                ave.append(extrema_t[n]-extrema_t[n-1])
            period = (np.mean(ave)*2)
            period_error = np.std(ave)*2
            self.frequency  = 1/(np.mean(ave)*2)
            self.frequency_error = period_error*period**(-2)
            print(sig_t)
            print(sig_peaks)
            print(len(sig_t))
            if len(sig_t)>1:
                popt,pcov = curve_fit(damping,sig_t,sig_peaks)
                self.damping_rate = popt[1]
                print(self.damping_rate)
                self.damping_error = np.sqrt(pcov[1][1])
                print(self.damping_error)
                print("damp")
        elif self.run_type == "TwoStream":
            print("this is a two stream")

    def plot(self):
        print(self.run_type)
        print(self.run_type == "TwoStream")
        if self.run_type == "Landau":
            extrema = scipy.signal.argrelextrema(np.array(self.s.firstharmonic),np.greater)[0]
            extrema = np.insert(extrema,0,0)
            extrema_t = []
            extrema_harm = []
            first_largest = -1
            sign = -1
            for n in extrema:
                extrema_t.append(self.s.t[n])
                extrema_harm.append(self.s.firstharmonic[n])
            for n in range (1,len(extrema_harm)-1):
                    if extrema_harm[n]>extrema_harm[n-1]:
                        first_largest = extrema[n]
                        sign = n
                        break
            extrema_harm = np.array(extrema_harm)
            extrema_t = np.array(extrema_t)
            noise_peaks = extrema_harm[sign:]
            noise_peak_t = extrema_t[sign:]
            sig  = np.array(self.s.firstharmonic[:first_largest])
            sig_time = np.array(self.s.t[:first_largest])
            noise = np.array(self.s.firstharmonic[first_largest:])
            noise_time = np.array(self.s.t[first_largest:])
            noise_average_sqaure = np.mean(noise**2)
            sig_average_sqaure = np.mean(sig**2)


            if noise_average_sqaure>0:
                self.noise_level = sig_average_sqaure/noise_average_sqaure
            sig_peaks = extrema_harm[:sign]
            sig_t = extrema_t[:sign]
            #calculating spacing between peaks
            ave = []
            for n in range (1,len(extrema_t)):
                ave.append(extrema_t[n]-extrema_t[n-1])
            period = (np.mean(ave)*2)
            period_error = np.std(ave)*2
            self.frequency  = 1/(np.mean(ave)*2)
            self.frequency_error = period_error*period**(-2)
            print("frequency = {}".format(self.frequency))
            print("frequency_error = {}".format(self.frequency_error))
            print("noise level = {}".format(self.noise_level))
            #print(self.damping_error)
            if len(sig_t)>1:
                popt,pcov = curve_fit(damping,sig_t,sig_peaks)
                self.damping_rate = popt[1]
            print(self.damping_rate)
            plt.figure()
            plt.plot(sig_time,sig)
            plt.plot(noise_time,noise)
            plt.scatter(sig_t, sig_peaks,marker="x")
            #plt.scatter(extrema_t, extrema_harm)
            plt.scatter(noise_peak_t,noise_peaks)
            plt.plot(sig_t,damping(sig_t,*popt))
            plt.xlabel("Time [Normalised]")
            plt.ylabel("First harmonic amplitude [Normalised]")
            plt.yscale('log')
            plt.ioff() # This so that the windows stay open
            plt.show()
        elif self.run_type == "TwoStream":
            print("making plot")
            extrema = scipy.signal.argrelextrema(np.array(self.s.firstharmonic),np.greater)[0]
            extrema = np.insert(extrema,0,0)
            extrema_t = []
            extrema_harm = []
            first_largest = -1
            sign = -1
            for n in extrema:
                extrema_t.append(self.s.t[n])
                extrema_harm.append(self.s.firstharmonic[n])
            for n in range (1,len(extrema_harm)-1):
                    if extrema_harm[n]>extrema_harm[n-1]:
                        first_largest = extrema[n]
                        sign = n
                        break
            extrema_harm = np.array(extrema_harm)
            extrema_t = np.array(extrema_t)
            popt,pcov = curve_fit(growth,extrema_t,extrema_harm,bounds=([0.,0.,-80.,0.], [1,1,0,1]))
            self.growth_rate = popt[0]*popt[1]
            plt.figure()
            plt.plot(self.s.t,self.s.firstharmonic)
            plt.scatter(extrema_t,extrema_harm)
            print(*popt)
            plt.plot(extrema_t,growth(np.array(extrema_t),*popt))
            plt.plot(extrema_t,self.growth_rate*(extrema_t+popt[2])+popt[3])
            plt.xlabel("Time [Normalised]")
            plt.ylabel("First harmonic amplitude [Normalised]")
            print("growth rate: {}".format(self.growth_rate))
            #plt.yscale()
            plt.ioff() # This so that the windows stay open
            plt.show()
            
    
 
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






particle_numbers = [20000,50000,100000,200000]
cell_numbers = [10,20,30,40,50,60,70,80,90,100]

print(particle_numbers)

def run_list():
    for npart in particle_numbers:
        for ncells in cell_numbers:
            print(npart)
            print(ncells)
            if Run_type == "TwoStream":
                # 2-stream instability
                L = 100
                ncells = 20
                pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
            elif Run_type == "Landau":
                # Landau damping
                L = 4.*pi
                ncells = 20
                pos, vel = landau(npart, L)

            s = Summary()                 # Calculates, stores and prints summary info

            diagnostics_to_run = [s]   # Remove p to get much faster code!
            # Run the simulation
            time_start = time.perf_counter()
            pos, vel = run(pos, vel, L, ncells, 
                        out = diagnostics_to_run,        # These are called each output step
                        output_times=linspace(0.,20,50)) # The times to output
            time_end= time.perf_counter()
            cal_time = time_end-time_start
            print("completed run, npart = {},ncells = {}, time taken = {}".format(npart,ncells,cal_time))
            obj = Run_Outcome(pos,vel,npart,ncells,cal_time,L,s)
            Save_name = "pn{}_cn{}".format(npart,ncells)
            save_object(obj,"data_{}/{}".format(Run_type,Save_name))

def Compare_runs():
    particle_numbers_loc = [5000,10000,20000,50000,100000,200000]
    cell_numbers_loc = [30,40,50,60,70,80,90,100]
    run_objs = []
    for npart in particle_numbers_loc:
        for ncells in cell_numbers_loc:
            Load_name = "pn{}_cn{}".format(npart,ncells)
            run_objs.append(load_object("data_{}/{}".format(Run_type,Save_name)))
    for obj in run_objs:
        obj.calculate_values()
    #plots at constant cell_number
    times_2d = []
    particle_number_2d = []
    frequencies_2d = []
    frequency_errors_2d = []
    noise_levels_2d = []
    damping_rates_2d = []
    damping_rates_errors_2d = []
    for ncells in cell_numbers_loc:
        times = []
        particle_number = []
        frequencies = []
        frequency_errors = []
        noise_levels = []
        damping_rates = []
        damping_rate_errors = []
        for obj in run_objs:
            if obj.ncells == ncells:
                times.append(obj.cal_time)
                particle_number.append(obj.npart)
                frequencies.append(obj.frequency)
                frequency_errors.append(obj.frequency_error)
                noise_levels.append(obj.noise_level)
                damping_rates.append(obj.damping_rate)
                damping_rate_errors.append(obj.damping_error)
        times_2d.append(times)
        particle_number_2d.append(particle_number)
        frequencies_2d.append(frequencies)
        frequency_errors_2d.append(frequency_errors)
        noise_levels_2d.append(noise_levels)
        damping_rates_2d.append(damping_rates)
        damping_rates_errors_2d.append(damping_rate_errors)
    #2time plots
    fig, axs = plt.subplots(2,2)
    axs[0,0]
    for n in range (0,len(cell_numbers_loc)):
        axs[0,0].plot(particle_number_2d[n],times_2d[n],label="cell number = {}".format(cell_numbers_loc[n]))
    axs[0,0].set_xlabel("Particle Number")
    axs[0,0].set_ylabel("Run Times (s)")

    #frequencies plots
    
    for n in range (0,len(cell_numbers_loc)):
        axs[0,1].plot(particle_number_2d[n],frequencies_2d[n])
        axs[0,1].fill_between(particle_number_2d[n],np.array(frequencies_2d[n])+np.array(frequency_errors_2d[n]),np.array(frequencies_2d[n])-np.array(frequency_errors_2d[n]),alpha=0.2)
    axs[0,1].set_xlabel("Particle Number")
    axs[0,1].set_ylabel("Frequencies (Hz)")
    axs[0,1].set_ylim(0.18, 0.24)

    #Noise Level plots
  
    for n in range (0,len(cell_numbers_loc)):
        axs[1,0].plot(particle_number_2d[n],noise_levels_2d[n])
    axs[1,0].set_xlabel("Particle Number")
    axs[1,0].set_ylabel("Noise Level")
   
    #damping rates plots
 
    for n in range (0,len(cell_numbers_loc)):
        axs[1,1].plot(particle_number_2d[n],damping_rates_2d[n])
        axs[1,1].fill_between(particle_number_2d[n], np.array(damping_rates_2d[n])-np.array(damping_rates_errors_2d[n]), np.array(damping_rates_2d[n])+np.array(damping_rates_errors_2d[n]),alpha=0.2)
    axs[1,1].set_xlabel("Particle Number")
    axs[1,1].set_ylabel("Damping Rate")
    fig.suptitle("Plots at constant Cell Number", fontsize=16)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc = (0.9, 0.8))
    plt.ioff() # This so that the windows stay open


    #plots at constant Particle number
    times_2d = []
    cell_number_2d = []
    frequencies_2d = []
    frequency_errors_2d = []
    noise_levels_2d = []
    damping_rates_2d = []
    damping_rates_errors_2d = []
    for npart in particle_numbers_loc:
        times = []
        cell_number = []
        frequencies = []
        frequency_errors = []
        noise_levels = []
        damping_rates = []
        damping_rate_errors = []
        for obj in run_objs:
            if obj.npart == npart:
                times.append(obj.cal_time)
                cell_number.append(obj.ncells)
                frequencies.append(obj.frequency)
                frequency_errors.append(obj.frequency_error)
                noise_levels.append(obj.noise_level)
                damping_rates.append(obj.damping_rate)
                damping_rate_errors.append(obj.damping_error)
        times_2d.append(times)
        cell_number_2d.append(cell_number)
        frequencies_2d.append(frequencies)
        frequency_errors_2d.append(frequency_errors)
        noise_levels_2d.append(noise_levels)
        damping_rates_2d.append(damping_rates)
        damping_rates_errors_2d.append(damping_rate_errors)
    #time plots
    fig, axs = plt.subplots(2,2)
    axs[0,0]
    for n in range (0,len(particle_numbers_loc)):
        axs[0,0].plot(cell_number_2d[n],times_2d[n],label="Particle number = {}".format(particle_numbers_loc[n]))
    #axs[0,0].legend()
    axs[0,0].set_xlabel("Cell Number")
    axs[0,0].set_ylabel("Run Time (s)")

    #frequencies plots
    
    for n in range (0,len(particle_numbers_loc)):
        axs[0,1].plot(cell_number_2d[n],frequencies_2d[n])
        axs[0,1].fill_between(cell_number_2d[n],np.array(frequencies_2d[n])+np.array(frequency_errors_2d[n]),np.array(frequencies_2d[n])-np.array(frequency_errors_2d[n]),alpha=0.2)
    #axs[0,1].legend()
    axs[0,1].set_xlabel("Cell Number")
    axs[0,1].set_ylabel("Frequency (Hz)")

    #Noise Level plots
  
    for n in range (0,len(particle_numbers_loc)):
        axs[1,0].plot(cell_number_2d[n],noise_levels_2d[n])
    #axs[1,0].legend()
    axs[1,0].set_xlabel("Cell Number")
    axs[1,0].set_ylabel("Noise Level")
   
    #Damping Rates plots
 
    for n in range (0,len(particle_numbers_loc)):
        axs[1,1].plot(cell_number_2d[n],damping_rates_2d[n])
        axs[1,1].fill_between(cell_number_2d[n], np.array(damping_rates_2d[n])-np.array(damping_rates_errors_2d[n]), np.array(damping_rates_2d[n])+np.array(damping_rates_errors_2d[n]),alpha=0.2)
    #axs[1,1].legend()
    axs[1,1].set_xlabel("Cell Numbers")
    axs[1,1].set_ylabel("Damping Rate")
    fig.suptitle("Plots at constant Particle Number", fontsize=16)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc = (0.85, 0.85))
    plt.ioff() # This so that the windows stay open
    plt.show()
    
    

if __name__ == "__main__":
    # Compare_runs()
    if Load == 0:
        # Generate initial condition
        npart = 50000   
        if Run_type == "TwoStream":
            # 2-stream instability
            L = 100
            ncells = 50
            pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
        elif Run_type == "Landau":
            # Landau damping
            L = 4.*pi
            ncells = 20
            pos, vel = landau(npart, L)
        # Create some output classes
        p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
        s = Summary()                 # Calculates, stores and prints summary info

        diagnostics_to_run = [s]   # Remove p to get much faster code!
        

        # Run the simulation
        time_start = time.perf_counter()
        pos, vel = run(pos, vel, L, ncells, 
                    out = diagnostics_to_run,        # These are called each output step
                    output_times=linspace(0.,80,200)) # The times to output
        time_end= time.perf_counter()
        cal_time = time_end-time_start
        obj = Run_Outcome(pos,vel,npart,ncells,cal_time,L,s,run_type=Run_type)
        save_object(obj,Save_name)
        obj.plot()
    elif Load == 1:
        obj = load_object(Load_name)
        pos,vel,ncells,L,s = obj.pos,obj.vel,obj.ncells,obj.L,obj.s
        p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
        obj.plot()