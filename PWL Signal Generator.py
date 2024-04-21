import numpy as np
import matplotlib.pyplot as plt

def generate_waveform(waveform, amplitude, frequency, duration, sampling_rate, f1=20, f2=20e3):
    t = np.arange(0, duration, 1 / sampling_rate)
    if(f1 <= 0):
        f1=1
    if waveform == 'triangle':
        waveform_data = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency)))
    elif waveform == 'sine':
        waveform_data = np.sin(2 * np.pi * frequency * t)
    elif waveform == 'square':
        waveform_data = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == 'chirp':
        L = (1/f1) * round(duration*f1 / (np.log(f2/f1)))
        waveform_data = np.sin(2 * np.pi * f1 * L * (np.exp(t/L)-1))
    return t, amplitude * waveform_data

def save_to_file(t, waveform_data, filename):
    with open(filename, 'w') as file:
        for time, voltage in zip(t, waveform_data):
            file.write(f"{time:.9e}\t{voltage:.9e}\n")

# Parameters
waveform_type = 'chirp'  # Choose 'triangle', 'sine', or 'square'
bBoth = False
user_input = input("Clean or Saturated? (c / s)").lower() 
while True:
    if user_input == 'c':
        bClean= True
        break
    elif user_input == 's':
        bClean = False
        break
    elif user_input == 'b':
        bClean = True
        bBoth = True
        break
    else:
        print("Invalid Input!")
        user_input = input("Clean or Saturated? (c / s)").lower() 

while True:        
    amplitude = 0.0001 if bClean else 1
    frequency = 500      
    duration = 1
    sampling_rate = 44.1e3 
    folder_path = 'C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\1. Generated Signals\\'
    if waveform_type != 'chirp':
        filename = waveform_type + '_' + str(amplitude) + 'V_' + str(frequency) + 'Hz_' + str(duration) + 's.txt'
    else:
        filename = waveform_type + '_' + str(amplitude) + 'V_' + str(duration) + 's.txt'
    output_path = folder_path + filename
    
    if bBoth:
        bClean = False
        bBoth = False
        
    else:
        break

# Generate waveform
time, voltage_data = generate_waveform(waveform_type, amplitude, frequency, duration, sampling_rate, 10, sampling_rate / 2 )
plt.figure(1)
plt.plot(time, voltage_data)
plt.show(block=False)
# Save to text file
save_to_file(time, voltage_data, output_path)
