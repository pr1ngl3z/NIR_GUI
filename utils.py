import numpy as np
from scipy.signal import savgol_filter

# Funktion zur Streuungskorrektur
def snv(input_data):
    output = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output

# Alternative Funktion zur Streuungskorrektur
def msc(input_data):
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
   
    # Durchschnittsspektrum wird als Referenzspektrum verwendet
    ref = np.mean(input_data, axis=0)
       
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
 
    return data_msc

def savgol():