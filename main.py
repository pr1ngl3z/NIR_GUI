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
        self.startWL = 930
        self.stopWL = 1692
        self.preset_startWL = self.startWL
        self.preset_stopWL = self.stopWL

        # configure window
        self.title("NIR Analyser")
        self.geometry(f"{860}x{720}")
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure((0, 1), weight=0)

        # Load Spectra via Button
        self.X = None
        self.y = None
        self.main_button_1 = customtkinter.CTkButton(master=self, text= 'LOAD Spectra', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.open_file_dialog)
        self.main_button_1.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Reset Button
        self.main_button_2 = customtkinter.CTkButton(master=self, text= 'RESET', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.reset)
        self.main_button_2.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Scale Down
        self.scale_frame = customtkinter.CTkFrame(self, width=140)
        self.scale_frame.grid(row=1, column=0, sticky='nsew')
        self.scale_label = customtkinter.CTkLabel(self.scale_frame, text='Specific WL', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.scale_label.grid(row=0, column=0, padx=20, pady=(20,10))
        
        self.scale_label1 = customtkinter.CTkLabel(self.scale_frame, text=f'Start WL (min {self.preset_startWL}):', font=customtkinter.CTkFont(size=14))
        self.scale_label1.grid(row=1, column=0, padx=(20,20), pady=(20, 0))
        self.entry_startWL = customtkinter.CTkEntry(master=self.scale_frame, placeholder_text=f'{self.preset_startWL}')
        self.entry_startWL.grid(row=2, column=0, padx=(20,20), pady=(10, 0), sticky="nsew")

        self.scale_label2 = customtkinter.CTkLabel(self.scale_frame, text=f'Stop WL (max {self.preset_stopWL}):', font=customtkinter.CTkFont(size=14))
        self.scale_label2.grid(row=3, column=0, padx=(20,20), pady=(20, 0))
        self.entry_stopWL = customtkinter.CTkEntry(master=self.scale_frame, placeholder_text=f'{self.preset_stopWL}')
        self.entry_stopWL.grid(row=4, column=0, padx=(20,20), pady=(10, 0), sticky="nsew")

        self.scale_button = customtkinter.CTkButton(master=self.scale_frame, text='Do Scaling', command=self.do_scaling)
        self.scale_button.grid(row=5, column=0, padx=20, pady=(20, 10))        
        
        # Preprocessing
        self.checkbox_frame = customtkinter.CTkFrame(self, width=140)
        self.checkbox_frame.grid(row=1, column=1, rowspan=3, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.checkbox_frame, text='Choose Preprocessing', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.check_savgol = customtkinter.BooleanVar(value=False)
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_frame, text='Savgol_Filter', variable=self.check_savgol, onvalue=True, offvalue=False)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20)

        self.check_snv = customtkinter.BooleanVar(value=False)
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_frame, text='SNV', variable=self.check_snv, onvalue=True, offvalue=False)
        self.checkbox_2.grid(row=2, column=0, pady=(20, 0), padx=20)

        self.preprocessing_button = customtkinter.CTkButton(master=self.checkbox_frame, text='Do Preprocessing', command=self.do_preprocessing)
        self.preprocessing_button.grid(row=3, column=0, padx=20, pady=(40, 10))

        # Data splitting
        self.splitting_frame = customtkinter.CTkFrame(self, width=140)
        self.splitting_frame.grid(row=1, column=2, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Data Splitting', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Random State = 42:', font=customtkinter.CTkFont(size=14))
        self.preprocessing_label.grid(row=1, column=0, padx=(20, 20), pady=(20, 0))
        self.entry_randstate = customtkinter.CTkEntry(master=self.splitting_frame, placeholder_text='42')
        self.entry_randstate.grid(row=2, column=0, padx=(20, 20), pady=(10, 0), sticky="nsew")

        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Test Size = 20 %:', font=customtkinter.CTkFont(size=14))
        self.preprocessing_label.grid(row=3, column=0, padx=(20, 20), pady=(20, 0))
        self.entry_split = customtkinter.CTkEntry(master=self.splitting_frame, placeholder_text='20')
        self.entry_split.grid(row=4, column=0, padx=(20, 20), pady=(10, 0), sticky="nsew")

        self.splitting_button = customtkinter.CTkButton(master=self.splitting_frame, text='Do Splitting', command=self.do_splitting)
        self.splitting_button.grid(row=5, column=0, padx=20, pady=(20, 10))
        
        # Regression
        self.regression_frame = customtkinter.CTkFrame(self, width=140)
        self.regression_frame.grid(row=1, column=3, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.regression_frame, text='Regression', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.radio_var = tk.IntVar(value=0)
        # self.label_radio_group = customtkinter.CTkLabel(master=self.regression_frame, text="Method:")
        # self.label_radio_group.grid(row=1, column=0, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.regression_frame, text='PLS', variable=self.radio_var, value=0)
        self.radio_button_1.grid(row=2, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.regression_frame, text='SVM', variable=self.radio_var, value=1)
        self.radio_button_2.grid(row=3, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.regression_frame, text='CNN', variable=self.radio_var, value=2)
        self.radio_button_3.grid(row=4, column=0, pady=10, padx=20, sticky="n")

        self.regression_button = customtkinter.CTkButton(master=self.regression_frame, text='Do Regression', command=self.do_regression)
        self.regression_button.grid(row=5, column=0, padx=20, pady=(40, 10))

        # Button config
        self.scale_button.configure(state='disabled')
        self.preprocessing_button.configure(state='disabled')
        self.splitting_button.configure(state='disabled')
        self.regression_button.configure(state='disabled')

        # Create a text box
        self.text_frame = customtkinter.CTkFrame(self, width=280)
        self.text_frame.grid(row=2, column=0, columnspan=4, sticky='nsew')

        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.columnconfigure(1, weight=1)
        self.text_frame.columnconfigure(2, weight=1)
        self.text_frame.columnconfigure(3, weight=1)

        self.text_label = customtkinter.CTkLabel(self.text_frame, text='Console Output:', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.text_label.grid(row=0, column=0, padx=(0,80), pady=(50, 0))
        self.text_box = customtkinter.CTkTextbox(self.text_frame)
        self.text_box.grid(row=1, column=0, columnspan=4, padx=(20, 20), pady=(20, 20), sticky="nsew")

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
        self.X_df = readX_and_y(folderpath)

        if len(self.X_df) > 1:
            print('SUCCESS Loading spectra')
            self.scale_button.configure(state='enabled')
        else:
            print('FAILED Loading Spectra')
        

    # Function for scaling
    def do_scaling(self):
        if len(self.entry_startWL.get()) > 1:
            self.preset_startWL = int(self.entry_startWL.get())
        # else:
        #     self.entry_startWL = self.preset_startWL
        print('Start WL = '+ str(self.preset_startWL))

        if len(self.entry_stopWL.get()) > 1:
            self.preset_stopWL = int(self.entry_stopWL.get())
        # else:
        #     self.entry_stopWL = self.preset_stopWL
        print('Stop WL = ' + str(self.preset_stopWL))

        self.X_df_specific = self.X_df.loc[:, self.preset_startWL:self.preset_stopWL]
        self.X = self.X_df_specific.to_numpy()
        self.y = np.array(self.X_df_specific.index)
        self.n_wavelenths = self.X.shape[1]
        print('Number of used wavelengths: '+ str(self.n_wavelenths))
        self.preprocessing_button.configure(state='enabled')
    
    # Preprocessing from Utils
    def do_preprocessing(self):
        if self.check_savgol.get() == True and self.check_snv.get() == False:
            if self.n_wavelenths > 25:
                self.X = savgol(self.X, 25, 2, deriv=1)
            else:
                self.X = savgol(self.X, self.n_wavelenths, 2, deriv=1)
            print('Savgol done')
        elif self.check_savgol.get() == False and self.check_snv.get() == True:
            self.X = snv(self.X)
            print('SNV done')
        elif self.check_savgol.get() == True and self.check_snv.get() == True:
            if self.n_wavelenths > 25:
                self.X = savgol(self.X, 25, 2, deriv=1)
            else:
                self.X = savgol(self.X, self.n_wavelenths, 2, deriv=1)
            self.X = snv(self.X)
            print('Savgol AND SNV done')
        else:
            self.X = self.X
            print('No Preprocessing done')
        print('SUCCESS Preprocessing')
        self.splitting_button.configure(state='enabled')


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
        self.regression_button.configure(state='enabled')
    
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
            if self.n_wavelenths > 80:
                parametersPLS = {'n_components': np.arange(1,80,1)}
            else:
                parametersPLS = {'n_components': np.arange(1,self.n_wavelenths,1)}
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
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv, self.X, self.n_wavelenths, self.preset_startWL, self.preset_stopWL)

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
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv, self.X, self.n_wavelenths, self.preset_startWL, self.preset_stopWL)

        # CNN
        elif self.radio_var.get() == 2:
            print('Starting CNN Regression')

            self.cnn_model = CNN(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test)

            self.X_train_scaled_row, self.X_val_scaled_row, self.X_test_scaled_row = self.cnn_model.standardize_row(self.X_train, self.X_val, self.X_test)
            self.X_train_scaled_rowcol, self.X_val_scaled_rowcol, self.X_test_scaled_rowcol = self.cnn_model.standardize_column(self.X_train_scaled_row, self.X_val_scaled_row, self.X_test_scaled_row)
            model = self.cnn_model.build_model()
            self.cnn_model.compile_model(model)

            self.progress_bar['maximum'] = self.cnn_model.EPOCHS

            class ProgressBarCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar):
                    self.progress_bar = progress_bar

                def on_epoch_end(self, epoch, logs=None):
                    self.progress_bar['value'] += 1
                    self.progress_bar.update()

            callback = [EarlyStopping(monitor='val_loss', patience=500, verbose=1), ProgressBarCallback(self.progress_bar)]

            fitted_model = self.cnn_model.fit_model(self.X_train_scaled_rowcol, self.y_train, self.X_val_scaled_rowcol, self.y_val, callback)

            self.progress_window.destroy()
            tf.keras.backend.clear_session()
            
            self.cnn_model.plot_loss(fitted_model)

            y_c = self.cnn_model.model.predict(self.X_train_scaled_rowcol)
            y_cv = self.cnn_model.model.predict(self.X_test_scaled_rowcol)
            y_vv = self.cnn_model.model.predict(self.X_val_scaled_rowcol)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print_metrics(score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv)
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_vv, score_cv, rmse_c, rmse_vv, rmse_cv, self.X, self.n_wavelenths, self.preset_startWL, self.preset_stopWL)
            
        else:
            print('No valid Choice!')

    # Function RESET    
    def reset(self):
        self.X = None
        self.y = None
        self.X_df = None
        self.X_df_specific = None
        self.n_wavelenths = None
        self.preset_startWL = self.startWL
        self.preset_stopWL = self.stopWL
        self.entry_startWL.delete(0, 'end')  
        self.entry_stopWL.delete(0, 'end')
        self.check_savgol.set(False)
        self.check_snv.set(False)      
        self.entry_randstate.delete(0, 'end')
        self.entry_split.delete(0, 'end')
        self.entry_split_answer = None
        self.random_state = None
        self.test_size = None
        self.radio_var.set(0)
        self.text_box.delete('1.0', tk.END)
        self.scale_button.configure(state='disabled')
        self.preprocessing_button.configure(state='disabled')
        self.splitting_button.configure(state='disabled')
        self.regression_button.configure(state='disabled')


if __name__ == "__main__":
    app = App()
    app.mainloop()