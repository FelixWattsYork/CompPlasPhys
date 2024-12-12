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
from joblib import Parallel, delayed
from itertools import product
import joblib
from scipy.stats import gaussian_kde
from scipy.integrate import solve_ivp

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


def calc_density_vec(position, ncells, L):

    nparticles = len(position)

    # Convert positions to cell indices
    p = (position*ncells) / L
    plower = np.floor(p).astype(int)  # Lower cell index (round down)
    offset = p - plower  # Fractional part (offset from the lower cell)

    # Distribute densities
    density = np.zeros(ncells, dtype=float)
    np.add.at(density, plower % ncells, 1.0 - offset)  # Add to the lower cell
    np.add.at(density, (plower + 1) % ncells, offset)  # Add to the upper cell

    # Normalize the density
    density *= float(ncells) / float(nparticles)
    return density

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
   

def pic(t,f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f) // 2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density_vec(pos, ncells, L)
    
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
    
    f_start = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    t_span  = [time,output_times[-1]]
    f = solve_ivp(pic,t_span,f_start,args = (ncells, L),t_eval=output_times)
    # print(pos)
    # print(vel)


   # Extract and process positions and velocities
    pos_all = ((f.y[:nparticles, :] % L) + L) % L  # Normalize positions for all times
    vel_all = f.y[nparticles:, :]                 # Velocities for all times

    # Call output functions for each time step
    for n, current_time in enumerate(output_times):
        for func in out:
            func(pos_all[:, n], vel_all[:, n], ncells, L, current_time)


    #     # Advance to tnext
    #     stepping = True
    #     while stepping:
    #         # Maximum distance a particle can move is one cell
    #         dt = cfl * dx / max(abs(vel))
    #         if time + dt >= tnext:
    #             # Next time will hit or exceed required output time
    #             stepping = False
    #             dt = tnext - time
    #         f = rk4step(pic, f, dt, args=(ncells, L))
    #         #f = solve_ivp(pic,dt,args = (ncells, L))
    #         time += dt to output functions
        # for func in out:
        #     func(pos, vel, ncells, L, time)
            
        #Extract position and velocities
        # print(f(tnext))
        # pos = ((f.y[0:nparticles] % L) + L) % L
        # vel = f.y[nparticles:]
        
        #Send to output functions
        # for func in out:
        #     func(pos, vel, ncells, L, time)
        
    return pos_all[:, -1], vel_all[:, -1]

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        
        d = calc_density_vec(pos, ncells, L)
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
        d = calc_density_vec(pos, ncells, L)
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
        d = calc_density_vec(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        #print(f"Time: {t} First: {fh}")
        
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
Save_name = "damping.pickle"
Load_name = "damping.pickle"
Run_type = "Landau"

import pickle
 
class Run_Outcome():
    def __init__(self,pos,vel,npart,ncells,cal_time,L,s,noise_level=0,frequency=0,frequency_error=0,damping_rate=0,damping_rate_error=0,run_type = "Unknown",growth_rate=0,growth_error=0,run_number = 0):
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
        self.growth_rate = growth_rate
        self.growth_error = growth_error
        self.run_number = run_number


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
            #print(sig_t)
            #print(sig_peaks)
            #print(len(sig_t))
            if len(sig_t)>1:
                popt,pcov = curve_fit(damping,sig_t,sig_peaks)
                self.damping_rate = popt[1]
                #print(self.damping_rate)
                self.damping_error = np.sqrt(pcov[1][1])
                #print(self.damping_error)
                #print("damp")
        elif self.run_type == "TwoStream":
            extrema = scipy.signal.argrelextrema(np.array(self.s.firstharmonic),np.greater)[0]
            extrema = np.insert(extrema,0,0)
            extrema_t = []
            extrema_harm = []
            first_largest = -1
            sign = -1
            for n in extrema:
                extrema_t.append(self.s.t[n])
                extrema_harm.append(self.s.firstharmonic[n])
            extrema_harm = np.array(extrema_harm)
            extrema_t = np.array(extrema_t)
            # splitting growth data from saturation data
            largest_index = np.argmax(extrema_harm)
            growth_data = extrema_harm[:largest_index]
            growth_t = extrema_t[:largest_index]

            if len(growth_t,)>1:
                try:
                    popt,pcov = curve_fit(growth,growth_t,growth_data,bounds=([0.,0.,-80.,0.], [1,1,0,1]))
                    perr = np.sqrt(np.diag(pcov))
                    self.growth_rate = popt[0]*popt[1]
                    self.growth_error = perr[0]*perr[1]
                except:
                    print("cannot fit data")

    def plot(self):
        #print(self.run_type)
        #print(self.run_type == "TwoStream")
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
            # un_numnercurrently I'm taking the mean of the noise peaks, however since this makes the noise level dependent on the time length I think this is only reasonable on 
            # integration methods that don't change kinetic energy, however since RK4 does this I'm going to make the noise level the amplitude of the first peak only 
            noise = np.array(self.s.firstharmonic[first_largest:])
            noise_time = np.array(self.s.t[first_largest:])
            noise_average_sqaure = np.mean(noise[0]**2)
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
            plt.axvline(x=noise_time[0], linestyle='dotted',color = "r",label = "Noise Dominated Region")
            plt.axhline(y=noise[0], linestyle='dotted', color = "orange",label = "Noise Level")
            plt.axhline(y=np.sqrt(sig_average_sqaure), linestyle='dotted', color = "blue",label = "Signal Level")
            plt.scatter(sig_t, sig_peaks,marker="x")
            #plt.scatter(extrema_t, extrema_harm)
            plt.scatter(noise_peak_t,noise_peaks)
            plt.plot(sig_t,damping(sig_t,*popt),label = "Wave Damping")
            plt.title("Plot of Harmonic Amplotude showing Damping of Landau wave")
            plt.xlabel("Time [Normalised]")
            plt.ylabel("First harmonic amplitude [Normalised]")
            plt.legend()
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
            extrema_harm = np.array(extrema_harm)
            extrema_t = np.array(extrema_t)
            # splitting growth data from saturation data
            largest_index = np.argmax(extrema_harm)
            growth_data = extrema_harm[:largest_index]
            growth_t = extrema_t[:largest_index]

            try:
                popt,pcov = curve_fit(growth,growth_t,growth_data,bounds=([0.,0.,-80.,0.], [1,1,0,1]))
                popt2,pcov2 = curve_fit(growth,extrema_t,extrema_harm,bounds=([0.,0.,-80.,0.], [1,1,0,1]))
                perr = np.sqrt(np.diag(pcov2))
                self.growth_rate = popt2[0]*popt2[1]
                self.growth_error = perr[0]*perr[1]
                # plt.figure()
                # plt.plot(self.s.t,self.s.firstharmonic)
                # plt.scatter(extrema_t,extrema_harm)
                # print(*popt)
                # plt.plot(extrema_t,growth(np.array(extrema_t),*popt2))
                # plt.plot(growth_t,self.growth_rate*(growth_t+popt2[2])+popt2[3])
                # plt.xlabel("Time [Normalised]")
                # plt.ylabel("First harmonic amplitude [Normalised]")
                # plt.ylim(0,max(extrema_harm)*1.1)
                # print("growth rate: {}".format(self.growth_rate))s
                # print("growth rate error: {}".format(self.growth_error))
                #plt.yscale()
                #plt.ioff() # This so that the windows stay open
                #plt.show()
            except:
                print("cannot fit data")
    
 
def save_object(obj,filename):
    print(obj)
    try:
        with open(filename, "wb") as f:
            joblib.dump(obj, f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return joblib.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)






particle_numbers = [20000]
cell_numbers = [20]
Run_Number = list(range(0, 6))
run_L = list(range(5,15,1))


def run_run(data):
    npart,ncells,Run,L_loc = data[0],data[1],data[2],data[3]
    random.seed(Run)
    if Run_type == "TwoStream":
        # 2-stream instability
        pos, vel = twostream(npart, L_loc, 3.) # Might require more npart than Landau!
        times_scale=linspace(0.,80,200)
    elif Run_type == "Landau":
        # Landau damping
        pos, vel = landau(npart, L_loc)
        times_scale=linspace(0.,40,100)

    s = Summary()                 # Calculates, stores and prints summary info

    diagnostics_to_run = [s]   # Remove p to get much faster code!
    # Run the simulation
    time_start = time.perf_counter()
    pos, vel = run(pos, vel, L_loc, ncells, 
                out = diagnostics_to_run,        # These are called each output step
                output_times = times_scale) # The times to output
    time_end= time.perf_counter()
    cal_time = time_end-time_start
    print("completed run, npart = {},ncells = {}, time taken = {}".format(npart,ncells,cal_time))
    obj = Run_Outcome(pos,vel,npart,ncells,cal_time,L_loc,s,run_type=Run_type,run_number=Run)
    Save_name = "pn{}_cn{}_{}_L_{}".format(npart,ncells,Run,L_loc)
    save_object(obj,"data_{}/{}".format(Run_type,Save_name))

def run_list():
    run_data = [list(comb) for comb in product(particle_numbers , cell_numbers , Run_Number,run_L)]
    #for data in run_data:
     #    run_run(data)
    Parallel(n_jobs=-1, backend='threading')(delayed(run_run)(data) for data in run_data)
                




def Compare_runs_TwoStream():
    particle_numbers_loc = [50000]
    cell_numbers_loc = [30]
    Run_Number_loc = list(range(1, 101))
    run_L_loc = [4.*pi]
    run_objs = []
    for npart in particle_numbers_loc:
        for ncells in cell_numbers_loc:
            for run in Run_Number_loc:
                Load_name = "pn{}_cn{}_{}_L_{}".format(npart,ncells,run,run_L_loc[1])
                run_objs.append(load_object("data_{}/{}".format(Run_type,Load_name)))
    growth_rates = []
    growth_rate_errors = []
    for obj in run_objs:
        #print(obj)
        obj.calculate_values()
        growth_rates.append(obj.growth_rate)
        growth_rate_errors.append(obj.growth_error)
        obj.plot()
    print(growth_rates)
    print(growth_rate_errors)
    print("growth rate average: {}".format(np.median(growth_rates)))
    # Estimate density
    growth_rates= [num for num in growth_rates if num <= 0.001]
    density = gaussian_kde(growth_rates)
    x = np.linspace(min(growth_rates), max(growth_rates), 1000)  # Points to evaluate the density
    y = density(x)

    # Plot the density
    plt.plot(x, y, label="Density")
    plt.fill_between(x, y, alpha=0.3)  # Optional: fill under the curve
    plt.xlabel("Value")
    #plt.xlim(0,0.001)
    plt.ylabel("Density")
    plt.title("Distribution Density Function")
    plt.legend()
    plt.show()

    
def Compare_runs():
    particle_numbers_loc = [1000,2000,5000,10000,20000,50000,100000,200000]
    #particle_numbers_loc = [50000]
    cell_numbers_loc = list(range(10, 210,10))
    Run_Number_loc = list(range(0, 6))
    run_L_loc = [4.*pi]
    run_objs = []
    for npart in particle_numbers_loc:
        for ncells in cell_numbers_loc:
            for run in Run_Number_loc:
                Load_name = "pn{}_cn{}_{}_L_{}".format(npart,ncells,run,run_L_loc[0])
                run_objs.append(load_object("data_{}/{}".format(Run_type,Load_name)))
                run_objs[-1].run_number = run
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
        for particle_num in particle_numbers_loc:
            times_av = []
            particle_number_av = []
            frequencies_av = []
            frequency_errors_av = []
            noise_levels_av = []
            damping_rates_av = []
            damping_rate_errors_av = []
            for run in Run_Number_loc:
                for obj in run_objs:
                    # print("comparison")
                    # print(obj.ncells)
                    # print(ncells)
                    # print(obj.npart)
                    # print(particle_num)
                    # print(obj.run_number)
                    # print(run)
                    # print(obj.ncells == ncells and obj.npart == particle_num and obj.run_number == run)
                    # print(obj.ncells == ncells)
                    # print(obj.npart == particle_num)
                    # print(obj.run_number == run)
                    if obj.ncells == ncells and obj.npart == particle_num and obj.run_number == run:
                        #print("true")
                        times_av.append(obj.cal_time)
                        particle_number_av.append(obj.npart)
                        frequencies_av.append(obj.frequency)
                        frequency_errors_av.append(obj.frequency_error)
                        noise_levels_av.append(obj.noise_level)
                        damping_rates_av.append(obj.damping_rate)
                        damping_rate_errors_av.append(obj.damping_error)
            times.append(np.mean(times_av))
            particle_number.append(np.mean(particle_number_av))
            frequencies.append(np.mean(frequencies_av))
            frequency_errors.append(np.mean(frequency_errors_av))
            noise_levels.append(np.mean(noise_levels_av))
            damping_rates.append(np.mean(damping_rates_av))
            damping_rate_errors.append(np.mean(damping_rate_errors_av))
            
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
        #axs[0,1].fill_between(particle_number_2d[n],np.array(frequencies_2d[n])+np.array(frequency_errors_2d[n]),np.array(frequencies_2d[n])-np.array(frequency_errors_2d[n]),alpha=0.2)
    axs[0,1].set_xlabel("Particle Number")
    axs[0,1].set_ylabel("Frequencies (Hz)")
    axs[0,1].set_ylim(0.18, 0.24)

    #Noise Level plots
  
    for n in range (0,len(cell_numbers_loc)):
        axs[1,0].plot(particle_number_2d[n],noise_levels_2d[n])
    axs[1,0].set_xlabel("Particle Number")
    axs[1,0].set_ylabel("Signal Strength")
   
    #damping rates plots
 
    for n in range (0,len(cell_numbers_loc)):
        axs[1,1].plot(particle_number_2d[n],damping_rates_2d[n])
        #axs[1,1].fill_between(particle_number_2d[n], np.array(damping_rates_2d[n])-np.array(damping_rates_errors_2d[n]), np.array(damping_rates_2d[n])+np.array(damping_rates_errors_2d[n]),alpha=0.2)
    axs[1,1].set_xlabel("Particle Number")
    axs[1,1].set_ylabel("Damping Rate")
    fig.suptitle("Plots at constant Cell Number", fontsize=16)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc = (0.9, 0.8))
    plt.ioff() # This so that the windows stay open


    #plots at constant Cell Number
    times_2d = []
    cell_number_2d = []
    frequencies_2d = []
    frequency_errors_2d = []
    noise_levels_2d = []
    damping_rates_2d = []
    damping_rates_errors_2d = []
    for nparticle in particle_numbers_loc:
        times = []
        cell_number = []
        frequencies = []
        frequency_errors = []
        noise_levels = []
        damping_rates = []
        damping_rate_errors = []
        for cell_num in cell_numbers_loc:
            times_av = []
            cell_number_av = []
            frequencies_av = []
            frequency_errors_av = []
            noise_levels_av = []
            damping_rates_av = []
            damping_rate_errors_av = []
            for run in Run_Number_loc:
                for obj in run_objs:
                    if obj.npart == nparticle and obj.ncells== cell_num and obj.run_number == run:
                        times_av.append(obj.cal_time)
                        cell_number_av.append(obj.ncells)
                        frequencies_av.append(obj.frequency)
                        frequency_errors_av.append(obj.frequency_error)
                        noise_levels_av.append(obj.noise_level)
                        damping_rates_av.append(obj.damping_rate)
                        damping_rate_errors_av.append(obj.damping_error)
            times.append(np.mean(times_av))
            cell_number.append(np.mean(cell_number_av))
            frequencies.append(np.mean(frequencies_av))
            frequency_errors.append(np.mean(frequency_errors_av))
            noise_levels.append(np.mean(noise_levels_av))
            damping_rates.append(np.mean(damping_rates_av))
            damping_rate_errors.append(np.mean(damping_rate_errors_av))
            
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
        #axs[0,1].fill_between(cell_number_2d[n],np.array(frequencies_2d[n])+np.array(frequency_errors_2d[n]),np.array(frequencies_2d[n])-np.array(frequency_errors_2d[n]),alpha=0.2)
    #axs[0,1].legend()
    axs[0,1].set_xlabel("Cell Number")
    axs[0,1].set_ylabel("Frequency (Hz)")

    #Noise Level plots
  
    for n in range (0,len(particle_numbers_loc)):
        axs[1,0].plot(cell_number_2d[n],noise_levels_2d[n])
    #axs[1,0].legend()
    axs[1,0].set_xlabel("Cell Number")
    axs[1,0].set_ylabel("Signal Strength")
   
    #Damping Rates plots
 
    for n in range (0,len(particle_numbers_loc)):
        axs[1,1].plot(cell_number_2d[n],damping_rates_2d[n])
        #axs[1,1].fill_between(cell_number_2d[n], np.array(damping_rates_2d[n])-np.array(damping_rates_errors_2d[n]), np.array(damping_rates_2d[n])+np.array(damping_rates_errors_2d[n]),alpha=0.2)
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
    #run_list()
    Compare_runs()
    # if Load == 0:
    #     random.seed(0)
    #     # Generate initial condition
    #     npart = 1000000
    #     if Run_type == "TwoStream":
    #         # 2-stream instability
    #         L = 100
    #         ncells = 50
    #         pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
    #     elif Run_type == "Landau":
    #         # Landau damping
    #         L = 4.*pi
    #         ncells = 100
    #         pos, vel = landau(npart, L)
    #     # Create some output classes
    #     p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    #     s = Summary()                 # Calculates, stores and prints summary info

    #     diagnostics_to_run = [s]   # Remove p to get much faster code!
        

    #     # Run the simulation
    #     time_start = time.perf_counter()
    #     pos, vel = run(pos, vel, L, ncells, 
    #                 out = diagnostics_to_run,        # These are called each output step
    #                 output_times=linspace(0.,80,200)) # The times to output
    #     time_end= time.perf_counter()
    #     cal_time = time_end-time_start
    #     obj = Run_Outcome(pos,vel,npart,ncells,cal_time,L,s,run_type=Run_type)
    #     print("cal_time{}".format(cal_time))
    #     save_object(obj,Save_name)
    #     obj.plot()
    # elif Load == 1:
    #     obj = load_object(Load_name)
    #     pos,vel,ncells,L,s = obj.pos,obj.vel,obj.ncells,obj.L,obj.s
    #     p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    #     obj.plot()