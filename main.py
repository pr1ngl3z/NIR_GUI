import os
import numpy as np
import tkinter as tk
from tkinter import ttk
import customtkinter
import sys
import random

from preprocessing import msc, snv, savgol
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal # type: ignore
from tensorflow.keras import layers # type: ignore
from keras.callbacks import EarlyStopping # type: ignore


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

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("NIR Analyser")
        self.geometry(f"{1100}x{580}")

        # Load Spectra via Button
        self.X = None
        self.y = None

        self.main_button_1 = customtkinter.CTkButton(master=self, text= 'LOAD Spectra', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.open_file_dialog)
        self.main_button_1.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Preprocessing
        self.checkbox_frame = customtkinter.CTkFrame(self, width=140)
        self.checkbox_frame.grid(row=1, column=0, rowspan=3, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.checkbox_frame, text='Choose Preprocessing', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.check_savgol = customtkinter.BooleanVar(value=False)
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_frame, text='Savgol_Filter', variable=self.check_savgol, onvalue=True, offvalue=False)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20)

        self.check_snv = customtkinter.BooleanVar(value=False)
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_frame, text='SNV', variable=self.check_snv, onvalue=True, offvalue=False)
        self.checkbox_2.grid(row=2, column=0, pady=(20, 0), padx=20)

        self.preprocessing_button = customtkinter.CTkButton(master=self.checkbox_frame, text='Do Preprocessing', command=self.do_preprocessing)
        self.preprocessing_button.grid(row=3, column=0, padx=20, pady=(20, 10))

        # Data splitting
        self.splitting_frame = customtkinter.CTkFrame(self, width=140)
        self.splitting_frame.grid(row=1, column=1, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Data Splitting', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Random State = 42:', font=customtkinter.CTkFont(size=14))
        self.preprocessing_label.grid(row=1, column=0, padx=20, pady=(20, 10))
        self.entry_randstate = customtkinter.CTkEntry(master=self.splitting_frame, placeholder_text='42')
        self.entry_randstate.grid(row=2, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Test Size = 20 %:', font=customtkinter.CTkFont(size=14))
        self.preprocessing_label.grid(row=3, column=0, padx=20, pady=(20, 10))
        self.entry_split = customtkinter.CTkEntry(master=self.splitting_frame, placeholder_text='20')
        self.entry_split.grid(row=4, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

        self.preprocessing_button = customtkinter.CTkButton(master=self.splitting_frame, text='Do Splitting', command=self.do_splitting)
        self.preprocessing_button.grid(row=5, column=0, padx=20, pady=(20, 10))

        # Regression
        self.regression_frame = customtkinter.CTkFrame(self, width=140)
        self.regression_frame.grid(row=1, column=2, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.regression_frame, text='Regression', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.radio_var = tk.IntVar(value=0)
        self.label_radio_group = customtkinter.CTkLabel(master=self.regression_frame, text="Method:")
        self.label_radio_group.grid(row=1, column=0, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.regression_frame, text='PLS', variable=self.radio_var, value=0)
        self.radio_button_1.grid(row=2, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.regression_frame, text='SVM', variable=self.radio_var, value=1)
        self.radio_button_2.grid(row=3, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.regression_frame, text='CNN', variable=self.radio_var, value=2)
        self.radio_button_3.grid(row=4, column=0, pady=10, padx=20, sticky="n")

        self.regression_button = customtkinter.CTkButton(master=self.regression_frame, text='Do Regression', command=self.do_regression)
        self.regression_button.grid(row=5, column=0, padx=20, pady=(20, 10))

        # Create a text box
        self.text_frame = customtkinter.CTkFrame(self, width=280)
        self.text_frame.grid(row=2, column=0, sticky='nsew')
        self.text_label = customtkinter.CTkLabel(self.text_frame, text='Console Output:', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.text_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.text_box = customtkinter.CTkTextbox(self.text_frame)
        self.text_box.grid(row=1, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        sys.stdout = self

    def write(self, txt):
        # Append the text to the text box
        self.text_box.insert('end', txt)
        self.text_box.update_idletasks

    def flush(self):
        # This could be used to ensure the text box is updated promptly, but in this case it does nothing
        pass

    # Function for choosing folder path and load spectra
    def open_file_dialog(self):
        folderpath = customtkinter.filedialog.askdirectory()
        print(folderpath)
        # Read spectra
        self.X, self.y = readX_and_y(folderpath)
        if len(self.X) > 1 and len(self.y) > 1:
            print('SUCCESS Loading spectra')
        else:
            print('FAILED Loading Spectra')

    # Preprocessing from Utils
    def do_preprocessing(self):
        if self.check_savgol.get() == True and self.check_snv.get() == False:
            self.X = savgol(self.X, 25, 2, deriv=1)
            print('Savgol done')
        elif self.check_savgol.get() == False and self.check_snv.get() == True:
            self.X = snv(self.X)
            print('SNV done')
        elif self.check_savgol.get() == True and self.check_snv.get() == True:
            self.X = savgol(self.X, 25, 2, deriv=1)
            self.X = snv(self.X)
            print('Savgol AND SNV done')
        else:
            self.X = self.X
            print('No Preprocessing done')
        print('SUCCESS Preprocessing')

    # Funtion for splitting data
    def do_splitting(self):
        if len(self.entry_randstate.get()) > 0:
            self.random_state = int(self.entry_randstate.get())
        else:
            self.random_state = 42
        print('Random state: ' + str(self.random_state))

        if len(self.entry_split.get()) > 0:
            self.entry_split_answer = float(self.entry_split.get())
            self.test_size = round((self.entry_split_answer/100.0),2)
        else:
            self.test_size = 0.2
        print('Test size: ' + str(self.test_size))

        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, random_state=self.random_state)
        print('SUCCESS Data splitting')

    # Function for regressions
    def do_regression(self):

        # Window for Progress bar
        self.progress_window = customtkinter.CTkToplevel(self)
        self.progress_window.title('Regression Progress')
        self.progress_bar = ttk.Progressbar(self.progress_window, length=500)
        self.progress_bar.pack()

        if self.radio_var.get() == 0:
            print('Starting PLS Regression')
            parametersPLS = {'n_components': np.arange(1,80,1)}
            pls = PLSRegression()
            cv = 10
            num_fits = cv * sum([len(v) for v in parametersPLS.values()])
            #progressbar = tqdm(total=num_fits, desc='Regression Progress')
            self.progress_bar['maximum'] = num_fits

            #update progress
            def scoringPLS(estimator, X, y):
                score = estimator.score(X,y)
                #progressbar.update()
                self.progress_bar['value'] += 1
                self.progress_bar.update()
                return score

            opt_pls = GridSearchCV(pls, parametersPLS, scoring=scoringPLS, verbose=0, cv=cv)
            opt_pls.fit(self.X_train, self.y_train)

            self.progress_window.destroy()

            print('Optimized Parameters: ')
            print(opt_pls.best_params_)

            pls = PLSRegression(n_components=opt_pls.best_params_['n_components'])
            pls.fit(self.X_train_val, self.y_train_val)

            y_c = pls.predict(self.X_train)
            y_cv = pls.predict(self.X_test)
            y_vv = pls.predict(self.X_val)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print("R2 calib: {:5.3f}".format(score_c))
            print("R2 val: {:5.3f}".format(score_vv))
            print("R2 test: {:5.3f}".format(score_cv))

            print("RMSE calib: {:5.3f}".format(rmse_c))
            print("RMSE val: {:5.3f}".format(rmse_vv))
            print("RMSE test: {:5.3f}".format(rmse_cv))

            z = np.polyfit(self.y_test, y_cv, 1) # gibt die Koeffizienten für mx+t aus, die am besten in die Punkte zw. Vorhersagewerte und tatsächliche Werte passt 
            with plt.style.context(("ggplot")):
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(y_cv, self.y_test, color = "red", edgecolor = "k")
                ax.plot(np.polyval(z,self.y_test), self.y_test, c = "blue", linewidth=1) # berechnete Koeffizienten z werden auf Daten in y_test angewendet und die entsprechenden y-Werte werden berechnet
                ax.plot(self.y_test, self.y_test, color = "green", linewidth=1)
                plt.title('PLS')
                plt.xlabel('Vorhersage Wassergehalt [%]')
                plt.ylabel('Tatsächlicher Wassergehalt [%]')
                legend_text='R² calib: {:.3f}\nR² val: {:.3f}\nR² test: {:.3f}\nRMSE calib: {:.3f}\nRMSE val: {:.3f}\nRMSE test: {:.3f}'.format(score_c,score_vv,score_cv ,rmse_c ,rmse_vv ,rmse_cv)
                ax.legend([legend_text] ,loc='lower right')
                plt.show() 

        elif self.radio_var.get() == 1:
            print('Starting SVM Regression')
            parametersSVM = {'C': [1, 100, 1000, 20000, 30000, 40000, 60000, 80000], \
                             'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 20, 50]}
            set_kernel = 'rbf'
            svm = SVR(kernel=set_kernel)
            cvSVM = 5
            # Code for Progress bar
            num_fitsSVM = cvSVM * len(parametersSVM['C']) * len(parametersSVM['gamma'])
            self.progress_bar['maximum'] = num_fitsSVM

            #update progress
            def scoringSVM(estimator, X, y):
                score = estimator.score(X,y)
                #progressbar.update()
                self.progress_bar['value'] += 1
                self.progress_bar.update()
                return score

            opt_svm = GridSearchCV(svm, parametersSVM, scoring=scoringSVM, verbose=0, cv=cvSVM)
            opt_svm.fit(self.X_train, self.y_train)

            self.progress_window.destroy()

            print('Optimized Parameters: ')
            print(opt_svm.best_params_)

            svm = SVR(kernel=set_kernel, C=opt_svm.best_params_['C'], gamma=opt_svm.best_params_['gamma'])
            svm.fit(self.X_train_val, self.y_train_val)

            y_c = svm.predict(self.X_train)
            y_cv = svm.predict(self.X_test)
            y_vv = svm.predict(self.X_val)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print("R2 calib: {:5.3f}".format(score_c))
            print("R2 val: {:5.3f}".format(score_vv))
            print("R2 test: {:5.3f}".format(score_cv))

            print("RMSE calib: {:5.3f}".format(rmse_c))
            print("RMSE val: {:5.3f}".format(rmse_vv))
            print("RMSE test: {:5.3f}".format(rmse_cv))

            z = np.polyfit(self.y_test, y_cv, 1) # gibt die Koeffizienten für mx+t aus, die am besten in die Punkte zw. Vorhersagewerte und tatsächliche Werte passt 
            with plt.style.context(("ggplot")):
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(y_cv, self.y_test, color = "red", edgecolor = "k")
                ax.plot(np.polyval(z,self.y_test), self.y_test, c = "blue", linewidth=1) # berechnete Koeffizienten z werden auf Daten in y_test angewendet und die entsprechenden y-Werte werden berechnet
                ax.plot(self.y_test, self.y_test, color = "green", linewidth=1)
                plt.title('SVM')
                plt.xlabel('Vorhersage Wassergehalt [%]')
                plt.ylabel('Tatsächlicher Wassergehalt [%]')
                legend_text='R² calib: {:.3f}\nR² val: {:.3f}\nR² test: {:.3f}\nRMSE calib: {:.3f}\nRMSE val: {:.3f}\nRMSE test: {:.3f}'.format(score_c,score_vv,score_cv ,rmse_c ,rmse_vv ,rmse_cv)
                ax.legend([legend_text] ,loc='lower right')
                plt.show() 

        elif self.radio_var.get() == 2:
            print('Starting CNN Regression')

            def standardize_row(X_train, X_val, X_test):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.T)
                X_val_scaled = scaler.fit_transform(X_val.T)
                X_test_scaled = scaler.fit_transform(X_test.T)
                return [X_train_scaled.T, X_val_scaled.T, X_test_scaled.T]

            def standardize_column(X_train, X_val, X_test):
                scaler = StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                return [X_train_scaled, X_val_scaled, X_test_scaled]
            
            def reproducible_comp():
                os.environ['PYTHONHASHSEED'] = '0'
                np.random.seed(42)
                random.seed(42)
                tf.random.set_seed(42)
                
            reproducible_comp() 
            
            X_train_scaled_col, X_val_scaled_col, X_test_scaled_col = standardize_column(self.X_train, self.X_val, self.X_test)
            X_train_scaled_row, X_val_scaled_row, X_test_scaled_row = standardize_row(self.X_train, self.X_val, self.X_test)
            X_train_scaled_rowcol, X_val_scaled_rowcol, X_test_scaled_rowcol = standardize_column(X_train_scaled_row, X_val_scaled_row, X_test_scaled_row)
            
            # Hypervariablen 
            INPUT_DIMS = np.shape(self.X_train)[1]
            CONV1D_DIMS = INPUT_DIMS
            K_NUMBER = 2 #1
            K_WIDTH = 5
            K_STRIDE = 1
            FC1_DIMS = 36
            FC2_DIMS = 18
            FC3_DIMS = 12
            OUT_DIMS = 1
            DROPOUT = 0.05
            EPOCHS = 10000
            BATCH = 1024
            LR_Adam = 0.005*BATCH/256. #0.01
            LR_RMS = 0.0005*BATCH/256. 

            self.progress_bar['maximum'] = EPOCHS

            # Variablen L2 Regularisierung
            beta = 0.003/2.
            K_REG = tf.keras.regularizers.l2(beta)

            # Initialisierung
            K_INIT = HeNormal(seed=42)
            model = tf.keras.Sequential([
                layers.Reshape((INPUT_DIMS, 1), input_shape=(INPUT_DIMS,)),
                layers.Conv1D(filters=K_NUMBER,
                            kernel_size=K_WIDTH,
                            strides=K_STRIDE,
                            padding='same', 
                            kernel_initializer=K_INIT,
                            kernel_regularizer=K_REG,
                            activation='elu',
                            input_shape=(CONV1D_DIMS,1)),
                layers.Dropout(DROPOUT),
                layers.Flatten(),
                layers.Dense(FC1_DIMS,
                            kernel_initializer=K_INIT,
                            kernel_regularizer=K_REG,
                            activation='elu'),
                layers.Dropout(DROPOUT),
                layers.Dense(FC2_DIMS,
                            kernel_initializer=K_INIT,
                            kernel_regularizer=K_REG,
                            activation='elu'),
                layers.Dropout(DROPOUT),
                layers.Dense(FC3_DIMS,
                            kernel_initializer=K_INIT,
                            kernel_regularizer=K_REG,
                            activation='elu'),
                layers.Dropout(DROPOUT),
                layers.Dense(1,
                            kernel_initializer=K_INIT,
                            kernel_regularizer=K_REG,
                            activation='linear')
                ])

            model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LR_RMS), loss='mse', metrics=['mse'])

            class ProgressBarCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar):
                    self.progress_bar = progress_bar

                def on_epoch_end(self, epoch, logs=None):
                    self.progress_bar['value'] += 1
                    self.progress_bar.update()

            callback = [EarlyStopping(monitor='val_loss', patience=500, verbose=1), ProgressBarCallback(self.progress_bar)]

            h1=model.fit(X_train_scaled_rowcol, self.y_train, batch_size=BATCH, epochs=EPOCHS, 
             validation_data=(X_val_scaled_rowcol, self.y_val),
             verbose=2,
             callbacks=callback)

            self.progress_window.destroy()
            tf.keras.backend.clear_session()
            
            with plt.style.context('ggplot'):
                plt.plot(h1.history['loss'], label='Training loss')
                plt.plot(h1.history['val_loss'], label='Validation loss')
                plt.yscale('log')
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.legend()
                plt.show()

            y_c = model.predict(X_train_scaled_rowcol)
            y_cv = model.predict(X_test_scaled_rowcol)
            y_vv = model.predict(X_val_scaled_rowcol)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print("R2 calib: {:5.3f}".format(score_c))
            print("R2 val: {:5.3f}".format(score_vv))
            print("R2 test: {:5.3f}".format(score_cv))

            print("RMSE calib: {:5.3f}".format(rmse_c))
            print("RMSE val: {:5.3f}".format(rmse_vv))
            print("RMSE test: {:5.3f}".format(rmse_cv))

            z = np.polyfit(self.y_test, y_cv, 1) # gibt die Koeffizienten für mx+t aus, die am besten in die Punkte zw. Vorhersagewerte und tatsächliche Werte passt 
            with plt.style.context(("ggplot")):
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(y_cv, self.y_test, color = "red", edgecolor = "k")
                ax.plot(np.polyval(z,self.y_test), self.y_test, c = "blue", linewidth=1) # berechnete Koeffizienten z werden auf Daten in y_test angewendet und die entsprechenden y-Werte werden berechnet
                ax.plot(self.y_test, self.y_test, color = "green", linewidth=1)
                plt.title(f'CNN (Dropout {DROPOUT})')
                plt.xlabel('Vorhersage Wassergehalt [%]')
                plt.ylabel('Tatsächlicher Wassergehalt [%]')
                legend_text='R² calib: {:.3f}\nR² val: {:.3f}\nR² test: {:.3f}\nRMSE calib: {:.3f}\nRMSE val: {:.3f}\nRMSE test: {:.3f}'.format(score_c,score_vv,score_cv ,rmse_c ,rmse_vv ,rmse_cv)
                ax.legend([legend_text] ,loc='lower right')
                plt.show()
            
        else:
            print('No valid Choice!')

        


if __name__ == "__main__":
    app = App()
    app.mainloop()