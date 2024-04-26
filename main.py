import os
import numpy as np
import tkinter as tk
from tkinter import ttk
import customtkinter
import sys

from preprocessing import msc, snv, savgol
from utils import readX_and_y, plot_metrics, print_metrics
from CNNclass import CNN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.svm import SVR

import tensorflow as tf
from keras.callbacks import EarlyStopping # type: ignore


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

        # PLS
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

            print_metrics(score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)

        # SVM
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

            print_metrics(score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)

        # CNN
        elif self.radio_var.get() == 2:
            print('Starting CNN Regression')

            cnn_model = CNN(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test)

            X_train_scaled_row, X_val_scaled_row, X_test_scaled_row = CNN.standardize_row(self.X_train, self.X_val, self.X_test)
            X_train_scaled_rowcol, X_val_scaled_rowcol, X_test_scaled_rowcol = CNN.standardize_column(X_train_scaled_row, X_val_scaled_row, X_test_scaled_row)

            cnn_model.compile_model()

            self.progress_bar['maximum'] = cnn_model.EPOCHS

            class ProgressBarCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar):
                    self.progress_bar = progress_bar

                def on_epoch_end(self, epoch, logs=None):
                    self.progress_bar['value'] += 1
                    self.progress_bar.update()

            callback = [EarlyStopping(monitor='val_loss', patience=500, verbose=1), ProgressBarCallback(self.progress_bar)]

            fitted_model = cnn_model.fit_model(X_train_scaled_rowcol, self.y_train, X_val_scaled_rowcol, self.y_val, callback)

            self.progress_window.destroy()
            tf.keras.backend.clear_session()
            
            cnn_model.plot_loss(fitted_model)

            y_c = cnn_model.model.predict(X_train_scaled_rowcol)
            y_cv = cnn_model.model.predict(X_test_scaled_rowcol)
            y_vv = cnn_model.model.predict(X_val_scaled_rowcol)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print_metrics(score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)
            
        else:
            print('No valid Choice!')

        


if __name__ == "__main__":
    app = App()
    app.mainloop()