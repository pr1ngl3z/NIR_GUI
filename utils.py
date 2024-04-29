import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funtion for reading spectra from folder
def readX_and_y(path):
    csvList = []
    for i in os.listdir(path):
        if i.endswith('.npy'):
            csvList.append(i)

    X = np.zeros((len(csvList), 256))
    y = np.zeros((len(csvList)))
    wl_int = np.linspace(930.033, 1852.05, 256, dtype=np.int16)

    i = 0
    for messung in csvList:
        y[i] = float(messung.split(sep='_')[1])/10.0
        data = np.load('{}/{}'.format(path, messung))
        X[i,:] = data[:]
        i = i + 1

    X_df = pd.DataFrame(X, columns=wl_int, index=y)
    return X_df

# Function for printing metrics
def print_metrics(score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv):
    print("R2 calib: {:5.3f}".format(score_c))
    print("R2 val: {:5.3f}".format(score_vv))
    print("R2 test: {:5.3f}".format(score_cv))

    print("RMSE calib: {:5.3f}".format(rmse_c))
    print("RMSE val: {:5.3f}".format(rmse_vv))
    print("RMSE test: {:5.3f}".format(rmse_cv))  

# Function for plotting metrics
def plot_metrics(radio_var, y_test, y_cv, score_c, score_vv, score_cv ,rmse_c ,rmse_vv ,rmse_cv, X, n_wavelength, start_wl, stop_wl):
    z = np.polyfit(y_test, y_cv, 1) # gibt die Koeffizienten für mx+t aus, die am besten in die Punkte zw. Vorhersagewerte und tatsächliche Werte passt 
    with plt.style.context(("ggplot")):
        _, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y_test, color = "red", edgecolor = "k", s=8)
        ax.plot(np.polyval(z,y_test), y_test, c = "blue", linewidth=1) # berechnete Koeffizienten z werden auf Daten in y_test angewendet und die entsprechenden y-Werte werden berechnet
        ax.plot(y_test, y_test, color = "green", linewidth=1)
        if radio_var == 0:
            plt.title(f'PLS (Start WL: {start_wl}, End WL: {stop_wl})')
        elif radio_var == 1:
            plt.title(f'SVM (Start WL: {start_wl}, End WL: {stop_wl})')
        elif radio_var == 2:
            plt.title(f'CNN (Start WL: {start_wl}, End WL: {stop_wl})')
        plt.xlabel('Predicted MC [%]')
        plt.ylabel('Measured MC [%]')
        plt.annotate(f'n wavelengths = {n_wavelength}\
                \nn spectra = {X.shape[0]}\
                \n60 % training, 20 % validation, 20 % test', 
                xy=(0, 1), xycoords='axes fraction', 
                xytext=(10, -10), textcoords='offset points', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plt.annotate('Metrics:\nR² train: {:.3f}\nR² val: {:.3f}\nR² test: {:.3f}\nRMSE train: {:.3f}\nRMSE val: {:.3f}\nRMSE test: {:.3f}'.format(score_c,score_vv,score_cv ,rmse_c ,rmse_vv ,rmse_cv), 
                    xy=(1, 0), xycoords='axes fraction', 
                    xytext=(-100, 10), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plt.show() 