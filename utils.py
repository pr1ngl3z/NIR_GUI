import os
import numpy as np
import matplotlib.pyplot as plt

# Funtion for reading spectra from folder
def readX_and_y(path):
    csvList = []
    for i in os.listdir(path):
        if i.endswith('.npy'):
            csvList.append(i)

    X = np.zeros((len(csvList), 256))
    y = np.zeros((len(csvList)))
    wl = np.linspace(930.033, 1852.05, 256)

    i = 0
    for messung in csvList:
        y[i] = float(messung.split(sep='_')[1])/10.0
        data = np.load('{}/{}'.format(path, messung))
        X[i,:] = data[:]
        i = i + 1

    return X, y

# Function for printing metrics
def print_metrics(score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv):
    print("R2 calib: {:5.3f}".format(score_c))
    print("R2 val: {:5.3f}".format(score_vv))
    print("R2 test: {:5.3f}".format(score_cv))

    print("RMSE calib: {:5.3f}".format(rmse_c))
    print("RMSE val: {:5.3f}".format(rmse_vv))
    print("RMSE test: {:5.3f}".format(rmse_cv))  

# Function for plotting metrics
def plot_metrics(radio_var, y_test, y_cv, score_c, score_vv, score_cv ,rmse_c ,rmse_vv ,rmse_cv):
    z = np.polyfit(y_test, y_cv, 1) # gibt die Koeffizienten für mx+t aus, die am besten in die Punkte zw. Vorhersagewerte und tatsächliche Werte passt 
    with plt.style.context(("ggplot")):
        _, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y_test, color = "red", edgecolor = "k")
        ax.plot(np.polyval(z,y_test), y_test, c = "blue", linewidth=1) # berechnete Koeffizienten z werden auf Daten in y_test angewendet und die entsprechenden y-Werte werden berechnet
        ax.plot(y_test, y_test, color = "green", linewidth=1)
        if radio_var == 0:
            plt.title('PLS')
        elif radio_var == 1:
            plt.title('SVM')
        elif radio_var == 2:
            plt.title('CNN')
        plt.xlabel('Vorhersage Wassergehalt [%]')
        plt.ylabel('Tatsächlicher Wassergehalt [%]')
        legend_text='R² calib: {:.3f}\nR² val: {:.3f}\nR² test: {:.3f}\nRMSE calib: {:.3f}\nRMSE val: {:.3f}\nRMSE test: {:.3f}'.format(score_c,score_vv,score_cv ,rmse_c ,rmse_vv ,rmse_cv)
        ax.legend([legend_text] ,loc='lower right')
        plt.show() 