# Imports
import pandas as pd
import sys
sys.path.append('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Python\\Data Handling')

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skopt import BayesSearchCV
from my_filter import *
from plots import *
import math
from sklearn.base import BaseEstimator
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
#from import_step_param import read_simulation_data

sys.path.append('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\Utils')
from myUtilities import read_simulation_data


# Global Definitions
function_type = 'atan'


# Functions

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def mapping_function_tanh(signal,kp,kn,gp,gn, kp2, kn2, gp2, gn2):
    upper_bound_condition = lambda x: x > kp
    lower_bound_condition = lambda x: x < -kn
    
    ap = (1-math.tanh(kp)**2)/gp
    bp = math.tanh(kp)
    an = (1-math.tanh(kn)**2)/gn
    bn=-math.tanh(kn)
  
    
    y = [ap*math.tanh(gp*(x-kp)) + (1-ap)*sigmoid(gp2*(x-kp2))+bp if upper_bound_condition(x)
         else an*math.tanh(gn*(x+kn))+ (1-an)*sigmoid(gn2*(x+kn2))+bn if lower_bound_condition(x)
         else math.tanh(x)
         for x in signal]
    
    return np.array(y)
    
def mapping_function_tanh_single(sample,params):
    kp, kn, gp, gn = params
    upper_bound_condition = lambda x: x > kp
    lower_bound_condition = lambda x: x < -kn
    
    ap = (1-math.tanh(kp)**2)/gp
    bp = math.tanh(kp)
    an = (1-math.tanh(kn)**2)/gn
    bn=-math.tanh(kn)
    
    y = [ap*math.tanh(gp*(x-kp))+bp if upper_bound_condition(x)
         else an*math.tanh(gn*(x+kn))+bn if lower_bound_condition(x)
         else math.tanh(x)
         for x in signal]
    
    return y

def mapping_function_atan(signal,kp,kn,gp,gn):
    upper_bound_condition = lambda x: x > kp
    lower_bound_condition = lambda x: x < -kn
    
    ap = (1-math.atan(kp)**2)/gp
    bp = math.atan(kp)
    an = (1-math.atan(kn)**2)/gn
    bn=-math.atan(kn)
    
    y = [ap*math.atan(gp*(x-kp))+bp if upper_bound_condition(x)
         else an*math.atan(gn*(x+kn))+bn if lower_bound_condition(x)
         else math.atan(x)
         for x in signal]
    
    return np.array(y)

def mapping_function_poly(signal,a1,a2,a3,a4):
    return a1+a2*signal+a3*signal**2+a4*signal**3

def mapping_function_poly8(signal,a0,a1,a2,a3,a4, a5,a6,a7):
    return a0+a1*signal+a2*signal**2+a3*signal**3+a4*signal**4+a5*signal**5+a6*signal**6+a7*signal**7


# Define your mathematical model function that takes 8 parameters
def model_system(g_pre, g_bias, g_wet, g_post, kp, kn, gp, gn, kp2, kn2, gp2, gn2):
    # Replace this with your system model
    # Example: y = a * x^2 + b * x + c * sin(d * x) + e * exp(f * x) + g * log(h * x)

    x_pre = h1Out * g_pre
    x_pre_mapping =  x_pre + x_bias*g_bias
    if function_type == 'tanh':
        x_m = mapping_function_tanh(x_pre_mapping,kp,kn,gp,gn, kp2, kn2, gp2, gn2)
    if function_type == 'atan':
        x_m = mapping_function_atan(x_pre_mapping,kp,kn,gp,gn)
    x_blend = x_m*g_wet + x_pre*(1-g_wet)
    nl_output = x_blend * g_post
    pred_output = signal.filtfilt(H2taps, 1, nl_output)
    
    return pred_output 


# Define the objective function to minimize
def scorer(g_pre, g_bias, g_wet, g_post, kp, kn, gp, gn, kp2, kn2, gp2, gn2):
    # Replace 'model_system' with your mathematical model function
    modeled_output = model_system(g_pre, g_bias, g_wet, g_post, kp, kn, gp, gn, kp2, kn2, gp2, gn2)
    # xcorr = signal.correlate(modeled_output, meas_output_normalized, 'full')

    # peaks,_=signal.find_peaks(xcorr)
    # max_peak_index = np.argmax(xcorr)

    # modeled_output=np.roll(modeled_output, max_peak_index)
    # # Calculate the mean squared error between modeled and measured output
    time_rmse = np.sqrt(np.mean((meas_output_normalized-modeled_output)**2))
    
    #fft_pred=np.fft.fft(modeled_output)
    #fft_meas=np.fft.fft(meas_output_normalized)
    frequencies, psd_pred = signal.welch(modeled_output, nperseg=len(modeled_output))
    frequencies, psd_meas = signal.welch(meas_output_normalized, nperseg=len(meas_output_normalized))
    freq_mse = mean_squared_error(psd_meas, psd_pred)
    
    err=time_rmse
    
    return -err

def read_filter_taps(filename):
    values = []
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read each line
        for line in file:
            # Append each line (stripped of whitespace) to the list
            values.append(line.strip())
            
    return values


#user_input = input("Clean or Saturated? (c / s)").lower() 
#while True:
#    if user_input == 'c':
#        bClean= True
#        break
#    elif user_input == 's':
#        bClean = False
#        break
#    else:
#        print("Invalid Input!")
#        user_input = input("Clean or Saturated? (c / s)").lower() 
#        
#        
#        
#filename = 'chirp_0.0001V_1s' if bClean else 'chirp_1V_1s'
#folder = 'Clean' if bClean else'Saturated'
#folder_path = 'C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\2. Target Signals\\LTSpice\\Time\\' + folder + '\\Resampled and Compensated\\'
#file_path = folder_path + filename + '.txt'
#df = read_simulation_data(file_path)
#input_signal = df['InputVoltage'] 
#output_signal = df['OutputVoltage']
#time = df['Time']
#plt.figure(1)
#plt.plot(time, input_signal)
#plt.plot(time, output_signal)
#plt.title('Input and Target Signal')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude (V)')
#plt.show(block=False)

path='C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\SPICE\\Simple SINE\\'

input_file='simple_sine_input_10m.txt'
#input_df = pd.read_csv(r'C:\Users\jhvaz\Documents\Faculdade\5º Ano\Tese\SPICE\Simple SINE\simple_sine_input_shrunk.txt', sep=',', header=None) simple sine
input_df = pd.read_csv(path+input_file, sep='\s+', header=1)
input_df.columns=['time', 'magnitude']

output_file='simple_sine_output_10m.txt'
#output_df = pd.read_csv(r'C:\Users\jhvaz\Documents\Faculdade\5º Ano\Tese\SPICE\Simple SINE\simple_sine_output_shrunk.txt', sep=',', header=None) simple sine
output_df = pd.read_csv(path+output_file, sep='\s+', header=1)
output_df.columns=['time', 'magnitude']

check_sync = input_df['time'].equals(output_df['time'])


# Load your measured input and output signals as NumPy arrays
time = input_df['time'].values
input_signal= input_df['magnitude'].values  # input data
input_signal_normalized=input_signal/max(input_signal)
meas_output = output_df['magnitude'].values  # measured output data
meas_output_normalized=meas_output/max(meas_output)

plot_comparison(1,time, input_signal_normalized, meas_output_normalized, 'Input Signal vs Output', "Time (ms)", 'Voltage (mV)', x_scale=1000, y_scale=1000)
bClean=False
#df = read_simulation_data(file_path)

fig=2
# ---------------------------------------------------------------- Data Loading and Normalization ----------------------------------------------------------------

#time = df['Time'].values
#input_signal= df['InputVoltage'].values  # input data
#meas_output = df['OutputVoltage'].values  # measured output data
#
#new_time = time
#meas_output_normalized = meas_output / max(meas_output)
#input_signal_normalized = input_signal / max(input_signal)

# ------------------------------------------------------------------------ Interpolation ----------------------------------------------------------------

sample_rate = 44.1e3

intFunctionInput = interp1d(time, input_signal_normalized)
intFunctionOutput = interp1d(time, meas_output_normalized, kind='cubic')

new_time = np.arange(1e-7, max(time), 1/sample_rate) 

resampled_input = intFunctionInput(new_time)
resampled_output = intFunctionOutput(new_time)

plot_comparison(fig, new_time, resampled_input, resampled_output, 'Input Signal vs Output after resampling' , "Time (ms)", 'Voltage (mV)', x_legend="Input", y_legend='Output', x_scale=1000, y_scale=1000)
fig=fig+1
input_signal_normalized_resampled=resampled_input
meas_output_normalized_resampled=resampled_output

# ------------------------------------------------------------------------ Interpolation ----------------------------------------------------------------

sample_rate = 44.1e3

intFunctionInput = interp1d(time, input_signal_normalized)
intFunctionOutput = interp1d(time, meas_output_normalized, kind='cubic')

new_time = np.arange(1e-7, max(time), 1/sample_rate) 

resampled_input = intFunctionInput(new_time)
resampled_output = intFunctionOutput(new_time)

plot_comparison(fig, new_time, resampled_input, resampled_output, 'Input Signal vs Output after resampling', "Time (ms)", 'Voltage (mV)', x_legend="Input", y_legend='Output', x_scale=1000, y_scale=1000)
fig=fig+1

input_signal_normalized_resampled=resampled_input
meas_output_normalized_resampled=resampled_output


# ------------------------------------------------------------------------ Phase Compensation ----------------------------------------------------------------

# calculate cross correlation of the two signals
xcorr = signal.correlate(input_signal_normalized_resampled, meas_output_normalized_resampled, 'full')

peaks,_=signal.find_peaks(xcorr)
max_peak_index = np.argmax(xcorr)
#phase_shift_samples=peaks[max_peak_index]

input_signal_normalized_resampled_phaseShifted=np.roll(input_signal_normalized_resampled, -max_peak_index)
input_signal_normalized_resampled_phaseShifted=input_signal_normalized_resampled_phaseShifted/max(input_signal_normalized_resampled_phaseShifted)

padding_length=max_peak_index
padding_value=0
#input_signal_normalized=np.pad(input_signal_normalized, (0, padding_length),mode='constant', constant_values=padding_value)


plot_comparison(fig, new_time, input_signal_normalized_resampled_phaseShifted, meas_output_normalized_resampled, 'Input Signal vs Output after phase shift', "Time (ms)", 'Voltage (mV)', x_legend="Input", y_legend='Output', x_scale=1000, y_scale=1000)
fig=fig+1

input_signal_normalized=input_signal_normalized_resampled_phaseShifted
meas_output_normalized=meas_output_normalized_resampled


# ------------------------------------------------------------------------ Low Pass Bias Filter Design ----------------------------------------------------------------

# Define the filter specifications
cutoff_frequency = 5.0  # Cutoff frequency in Hz
sampling_frequency = 10000  # Sampling frequency in Hz
filter_order = 4 # Filter order (adjust as needed)
num_taps=211

# Design the low-pass filter
b, a = design_BES_LP_filter(cutoff_frequency, sampling_frequency, filter_order)
#b=design_fir_LP_filter(cutoff_frequency, num_taps, sampling_frequency)
#b = design_fir_LP_filter(cutoff_frequency, num_taps, sampling_frequency)
# Plot the frequency response (magnitude and phase)
plot_frequency_response(b,a, sampling_frequency)


x_bias = signal.filtfilt(b, a, abs(input_signal_normalized))

plot_comparison(fig, new_time, input_signal_normalized, x_bias, 'Input Signal vs Bias', "Time (ms)", 'Voltage (mV)', x_legend='Input Signal', y_legend="Bias", x_scale=1000, y_scale=1000)
fig=fig+1
plot_comparison(fig, new_time, input_signal_normalized, meas_output_normalized, 'Input Signal vs Output', "Time (ms)", 'Voltage (mV)', x_legend="Input", y_legend='Output', x_scale=1000, y_scale=1000)
fig=fig+1


# ------------------------------------------------------------------------ H1 -----------------------------------------------------------------------------------

folder_path = 'C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\'
nTaps = 199
voltage = '0.0001V' if bClean else "1V"
filter_type = 'FIR'
filename = 'H1_' + filter_type + '_' + str(nTaps) + ' taps_freq_chirp_' + str(voltage) + '_1s'
file_path = folder_path + filename + '.txt'

H1taps_raw = read_filter_taps(file_path)
H1taps = [float(x) for x in H1taps_raw]
H1taps = [x/max(H1taps) for x in H1taps]

h1Out = signal.filtfilt(H1taps, 1, input_signal_normalized)
plot_comparison(fig, new_time, input_signal_normalized, h1Out, 'H1 Stage', "Time (ms)", 'Voltage (mV)', x_legend='Input Signal', y_legend="Output Signal", x_scale=1000, y_scale=1000)
fig=fig+1

plot_comparison(fig, new_time, meas_output_normalized, h1Out, 'H1 Stage vs Measured Output', "Time (ms)", 'Voltage (mV)', x_legend='Meas Output', y_legend="H1", x_scale=1000, y_scale=1000)
fig=fig+1
# ------------------------------------------------------------------------ H2 -----------------------------------------------------------------------------------

filename = 'H2_' + filter_type + '_' + str(nTaps) + ' taps_freq_chirp_' + str(voltage) + '_1s'
file_path = folder_path + filename + '.txt'

H2taps_raw = read_filter_taps(file_path)
H2taps = [float(x) for x in H2taps_raw]
H2taps = [x/max(H2taps) for x in H2taps]

# ------------------------------------------------------------------------ Bayesian Optimization ----------------------------------------------------------------

pre_max=1

#param_space = {
#    'g_pre': (0.1, 1),
#    'g_bias':(0, 0),
#    'g_wet': (0, 1),
#    'g_post':(0.1, 1),
#    'a0':(-pre_max, pre_max),
#    'a1':(-pre_max, pre_max),
#    'a2':(-pre_max, pre_max),
#    'a3':(-pre_max, pre_max),
#    'a4':(-pre_max, pre_max),
#    'a5':(-pre_max, pre_max),
#    'a6':(-pre_max, pre_max),
#    'a7':(-pre_max, pre_max)
#    }

param_space = {
    'g_pre': (0.1, 100),
    'g_bias':(-1, 1),
    'g_wet': (0.8, 1),
    'g_post':(0.1, 100),
    'kp':(0, 1),
    'kn':(-1, 0),
    'gp':(0.01, 10),
    'gn':(0.01, 10),
    'kp2':(0, 1),
    'kn2':(-1, 0),
    'gp2':(0.01, 10),
    'gn2':(0.01, 10)
    }

# Initialize the Bayesian optimization algorithm.
optimizer = BayesianOptimization(
f=scorer,
pbounds=param_space,  # The search space of the parameters.
random_state=42
)

# Run the Bayesian optimization algorithm.
initPoints = 150
nIter=2000
optimizer.maximize(init_points=initPoints, n_iter=nIter)

# Get the best set of parameters.
res_dict = optimizer.max
print(res_dict)
params_dict = res_dict['params']

res_dict = optimizer.max
print(res_dict)
params_dict = res_dict['params']

g_pre=params_dict['g_pre']
g_bias=params_dict['g_bias']
g_wet=params_dict['g_wet']
g_post=params_dict['g_post']
gp = round(params_dict['gp'], 3)
gn = round(params_dict['gn'], 3)
kp = round(params_dict['kp'], 3)
kn = round(params_dict['kn'], 3) 
gp2 = round(params_dict['gp2'], 3)
gn2 = round(params_dict['gn2'], 3)
kp2 = round(params_dict['kp2'], 3)
kn2 = round(params_dict['kn2'], 3)
best_pred=model_system(g_pre, g_bias, g_wet, g_post, kp, kn, gp, gn, kp2, kn2, gp2, gn2)
plot_comparison(fig, new_time, meas_output_normalized, best_pred, 'Measured Output vs Model Prediction', "Time (ms)", 'Voltage (mV)', x_legend="Target", y_legend="Prediction", x_scale=1000, y_scale=1000)
fig=fig+1
error=-res_dict['target']
rmse_percentage = error/(np.max(meas_output_normalized)-np.min(meas_output_normalized)) * 100

# Get the best set of parameters.
plt.figure(fig)
plt.plot(new_time, best_pred)

plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.legend()
plt.savefig("C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\4. Wiener Hammerstein Model Optimization\\Plots\\plot.png", dpi=300)
plt.show()
fig = fig+1

# ------------------------------------------------------------------------ Data Export ----------------------------------------------------------------

# Specify the file path where you want to save the variables
# Specify the file path where you want to save the variables
export_path="C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\Python\\NL Block\\Registos\\"
export_file='NLBlock_parameters_' + function_type + '_errorInPercentage.txt'
file_path = export_path + export_file
error=round(error,5)
rmse_percentage=round(rmse_percentage,5)
g_pre = round(g_pre, 3)
g_bias = round(g_bias, 3)
g_wet = round(g_wet, 3)
g_post = round(g_post, 3)

# Open the file for appending (creates the file if it doesn't exist)
with open(file_path, "a") as file:
    # If the file is empty, write column headers and separator line
    if file.tell() == 0:
        column_names = ['rmse (%)', 'g_pre', 'g_bias','g_wet','g_post','gp', 'gn', 'kp', 'kn', 'init_points', 'n_iter']
        row_str=f"{column_names[0]:<10} | "
        for i in range(1,len(column_names)):
            row_str+=f"{column_names[i]:<10} | "
        row_str+='\n'
        file.write(row_str)
        file.write("-" * (13*len(column_names)-1) + "\n")

    # Write the variables to the file in a tabular format
    file.write(f"{rmse_percentage:<10} | {g_pre:<10} | {g_bias:<10} | {g_wet:<10} | {g_post:<10} | {gp:<10} | {gn:<10} | {kp:<10} | {kn:<10} | {initPoints:<10} | {nIter:<10} |\n")

# Confirm that the variables have been exported
print("Parameters exported to", file_path)

rmse=error

print("RMSE: ", rmse)
print("RMSE Percentage: ", rmse_percentage)


plot_comparison(fig, new_time, input_signal_normalized, best_pred, 'Measured Output vs Model Prediction', "Time (ms)", 'Voltage (mV)', x_legend="Target", y_legend="Prediction", x_scale=1000, y_scale=1000)