# bvd_approximation
This python scripts uses a genetic algorithm to find the parameters for an extended butterworth van dyke model to approximate a given impedance response

# How To Use
Convert your measured impedance response you want to approximate into a .npy file (numpy saved array file)
Place that npy file in the same folder as  bvd_optimization_script.py
Open command prompt in that folder
Type python bvd_optimization_script.py
Answer questions that are given
The log of the parameters will be saved in that folder
The parameters are in the following order, first the lone capacitor, then pairs of 3 values where it is capacitor, resistor, inductor in that order
