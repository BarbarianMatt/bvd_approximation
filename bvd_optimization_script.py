import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import differential_evolution
from datetime import datetime

# this converts the user input to a number
def input_to_number(user_input):
    try:
        x=float(user_input)
        return x
    except ValueError:
        raise Exception("Invalid Input")

# this converts the user input to a filepath
def input_to_filepath(user_input):
    if os.path.isfile(user_input):
        return user_input
    else:
        raise Exception("Invalid Input")

# this get the user inputs and converts them to usable parameters
def get_parameters():
    minorfile = input("Enter Min Frequency (if non-uniform frequency, type filepath to frequency vector as a .npy file): ")
    if os.path.isfile(minorfile):
        file=input_to_filepath(minorfile)
        freq=np.load(file)
    else:
        min=input_to_number(minorfile)
        max=input_to_number(input("Enter Max Frequency: "))
        input1=input("Sample Rate (if you start your input with an 'n' it will be number of samples instead): ")
        if input1[0]=='n':
            samples=int(input_to_number(input1[1:]))
            freq=np.linspace(min, max, samples)
        else:
            fs=input_to_number(input1)
            freq=np.arange(min, max, 1/fs)

    md_filepath=input_to_filepath(input("Enter Filepath to Measure Impedance Response: "))
    measured_data=np.load(md_filepath)

    rlc_num=int(input_to_number(input("Number of RLC Paths in Extended BVD Model: ")))
    
    

    return freq,measured_data,rlc_num

# this a wrapper function for the impedance calc function
def impedance_calc_wrapper(parameters, f):
        c0 = parameters[0]
        cN=parameters[1::3]
        rN=parameters[2::3]
        lN=parameters[3::3]
        final = impedance_calc_butterworthvandyke(f, rN, lN, cN, c0)
        return final

# this function calculates the impedance response of an expanded butterworth van dyke model
def impedance_calc_butterworthvandyke(omega, r_array, l_array, c_array, c_0):
        Z_c0 = 1j*omega*c_0
        Z_m= r_array[:,None]*np.ones_like(omega)[None,:] + l_array[:,None] * 1j * omega[None,:] + 1/(c_array[:,None] * 1j * omega[None,:])
        output=np.sum(1/Z_m,axis=0)+Z_c0
        return 1/output

# this function runs the genertic algorthim to approximate the measured impedance response with the bvd model using rlc_num number of resonances
def genetic_algorithm_approximation(freq,measured_impedance,rlc_num):
    def ge_objective(params, Z, omega):
        Z_calculated = impedance_calc_wrapper(params, omega)
        error = np.sum((np.abs(Z_calculated - Z)*1e3)**2)
        return error


    
    bounds = [(-19, -1)] # arbitrary bounds I found to work well, can be changed
    bounds += [(-17, -1),(-11, 5),(-17, -1)] * rlc_num

    pop_size=80
    max_iter=5e3
    tolerance=1e-7

    omega = 2 * np.pi * freq
    comparison=measured_impedance

    def wrapped_objective(params):
        return ge_objective(10**params, comparison, omega)
    
    
    optimized_params = differential_evolution(wrapped_objective, bounds, popsize=int(pop_size), maxiter=int(max_iter), tol=tolerance)
    print(optimized_params)
    final_parameters=np.array(optimized_params.x)

    np.save('optimized_parameters.npy',final_parameters)
    print("log() of optimized parameters saved as optimized_parameters.npy")

    return final_parameters

# this function plots the measured impedance response and the approximated one
def plot_values(freq,measured_data,optimized):
    omega = 2 * np.pi * freq

    approximation=impedance_calc_wrapper(10**optimized,omega)

    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(measured_data), color='orange',label='Measured')
    plt.plot(freq, np.abs(approximation), color='purple',label='Approximation')
    plt.legend()
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Impedance')
    plt.title('Impedance Measured vs Approximation')
    plt.show()

# this version of the program gets the parameters from user input
def user_input_main():
    # current_time = datetime.now().time()
    # print("Current Time:", current_time)

    freq,measured_data,rlc_num = get_parameters()

    final_parameters=genetic_algorithm_approximation(freq,measured_data,rlc_num)

    # current_time = datetime.now().time()
    # print("Current Time:", current_time)

    plot_values(freq,measured_data,final_parameters)

# this version of the program gets the parameters from code input
def code_input_main():
    # current_time = datetime.now().time()
    # print("Current Time:", current_time)

    freq=np.linspace(1e6, 8e6, 1601)
    measured_data=np.load("measured1.npy")
    rlc_num=6

    final_parameters=genetic_algorithm_approximation(freq,measured_data,rlc_num)

    # current_time = datetime.now().time()
    # print("Current Time:", current_time)

    plot_values(freq,measured_data,final_parameters)

# this is main
if __name__ == "__main__":
    user_input_main()
    # code_input_main()
