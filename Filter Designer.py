import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter, butter, firls, firwin2, remez
import numpy as np
import sys
sys.path.append('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\Utils')
from myUtilities import read_simulation_data


def getFilename(mode, duration):
    if mode == 'clean':
        voltage = 0.0001
    else:
        voltage = 1
    filename = 'freq_chirp_' + str(voltage) + 'V_' + str(duration) + 's' 
    return filename

filename = 'freq_chirp_1V_1s'
filename = getFilename('clean', 1)
folder_path = 'C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\2. Target Signals\\LTSpice\\Frequency\\' 
file_path = folder_path + filename + '.txt'
output_folder_path = 'C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\Plots\\'

plot_name = filename + "_Htotal.png"
df = read_simulation_data(file_path)
input = df['InputMagnitude'] 
output = df['OutputMagnitude']
frequencies = df['Frequency']
htotal = output / input
plt.figure(1)
plt.semilogx(frequencies, 20 * np.log10(input), label='Input')
plt.semilogx(frequencies, 20 * np.log10(output), label='Output')
plt.semilogx(frequencies, 20 * np.log10(htotal), label='H Total')
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.savefig(output_folder_path + plot_name, dpi=300)
plt.show(block=False)

plot_name = filename + "_H2.png"
filename = getFilename('saturated', 1)
folder_path = 'C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\2. Target Signals\\LTSpice\\Frequency\\' 
file_path = folder_path + filename + '.txt'
df = read_simulation_data(file_path)
input = df['InputMagnitude'] 
output = df['OutputMagnitude']
frequencies = df['Frequency']
h2 = output / input
plt.figure()
plt.semilogx(frequencies, 20 * np.log10(input), label='Input')
plt.semilogx(frequencies, 20 * np.log10(output), label='Output')
plt.semilogx(frequencies, 20 * np.log10(h2), label='H2')
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.savefig(output_folder_path + plot_name, dpi=300)
plt.legend()
plt.grid(True)
plt.show(block=False)


plot_name = filename + "_H1.png"
h2_div = h2
increment_value = 1e-9
h2_div[h2_div == 0] += increment_value
h1 = htotal / h2_div
print(h1)
print(h2)
print(htotal)
plt.figure()
plt.semilogx(frequencies, 20 * np.log10(h1), label='H1')
plt.semilogx(frequencies, 20 * np.log10(h2), label='H2')
plt.semilogx(frequencies, 20 * np.log10(htotal), label='H Total')
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.savefig(output_folder_path + plot_name, dpi=300)
plt.grid(True)
plt.show(block=False)

#FIR

responses = [h1, h2]
count = 0
for r in responses:
    fs = 44.1e3
    num_taps = 199
    norm_frequencies = np.pad(frequencies, (1, 1), mode='constant', constant_values=(0, fs/2))
    desired_frequency_response = np.pad(r, (1, 1), mode='constant', constant_values=(0, 0))
    print(desired_frequency_response)
    f= np.array(abs(norm_frequencies / max(norm_frequencies)))
    m= np.array(abs(desired_frequency_response))

    print(f)
    print(m)
    b = firwin2(num_taps, f, m)
    a = 1    

    if count == 0:
        np.savetxt('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\H1_FIR_' + str(num_taps) + ' taps_' + filename + '.txt', b)
    else:
        np.savetxt('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\H2_FIR_' + str(num_taps) + ' taps_' + filename + '.txt', b)
    # Frequency response of the designed IIR filter
    w, h = freqz(b, a, fs=fs)

    if count == 0:
        np.savetxt('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\Response\\H1_' + filename + '.txt', (f,m))
    else:
        np.savetxt('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\Response\\H2_' + filename + '.txt', (f,m))

    # Desired frequency response
    
    if count == 0:
        output_filename = 'H1_' + filename + '_' + str(num_taps) + " taps_FIR_response.png"
    else:
        output_filename = 'H2_' + filename + '_' + str(num_taps) + " taps_FIR_response.png"
        
    plt.figure(count + 4)
    plt.semilogx(norm_frequencies, 20 * np.log10(np.abs(m)), label='Desired Response', linestyle='dashed')
    plt.semilogx(w, 20 * np.log10(np.abs(h)), label='FIR Filter')
    plt.title('H1 - FIR Filter Frequency Response' if count==0 else 'H2 - FIR Filter Frequency Response')
    plt.xlabel('Frequency (radians)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.savefig(output_folder_path + output_filename, dpi=300)
    plt.show(block = False if count == 0 else True)
    
    count+=1
