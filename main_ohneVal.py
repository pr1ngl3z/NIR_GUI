import numpy as np
import tkinter as tk
from tkinter import ttk
import customtkinter
import sys
import seaborn as sns

from preprocessing import msc, snv, savgol
from utils_ohneVal import readX_and_y, plot_metrics, print_metrics, plot_samples
from CNNclass import CNN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping # type: ignore
import pickle


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
        self.geometry(f"{1300}x{550}")

        # Load Spectra via Button
        self.X = None
        self.y = None
        self.main_button_1 = customtkinter.CTkButton(master=self, text= 'LOAD Spectra', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.open_file_dialog)
        self.main_button_1.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Reset Button
        self.main_button_2 = customtkinter.CTkButton(master=self, text= 'RESET', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.reset)
        self.main_button_2.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Scale Down
        self.var_startWL = tk.IntVar(value=930)
        self.var_stopWL = tk.IntVar(value=1692)

        self.scale_frame = customtkinter.CTkFrame(self, width=140)
        self.scale_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.scale_label = customtkinter.CTkLabel(self.scale_frame, text='Specific WL', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.scale_label.grid(row=0, column=0, padx=20, pady=(20,10))
        
        self.scale_label1 = customtkinter.CTkLabel(self.scale_frame, text=f'Start WL (min {self.preset_startWL}):', font=customtkinter.CTkFont(size=12))
        self.scale_label1.grid(row=1, column=0, padx=20, pady=(20,5), sticky='w')
        self.slider_startWL = customtkinter.CTkSlider(self.scale_frame, variable=self.var_startWL, from_=930, to=1692, number_of_steps=762, command=self.update_startWL)
        self.slider_startWL.grid(row=2, column=0, padx=(10, 10), pady=(5, 5), sticky="e")
        self.update_startWL(self.slider_startWL.get())

        self.scale_label2 = customtkinter.CTkLabel(self.scale_frame, text=f'Stop WL (max {self.preset_stopWL}):', font=customtkinter.CTkFont(size=12))
        self.scale_label2.grid(row=3, column=0, padx=20, pady=(10,5), sticky='w')
        self.slider_stopWL = customtkinter.CTkSlider(self.scale_frame, variable=self.var_stopWL, from_=930, to=1692, number_of_steps=762, command=self.update_stopWL)
        self.slider_stopWL.grid(row=4, column=0, padx=(10, 10), pady=(5, 5), sticky="e")
        self.update_stopWL(self.slider_stopWL.get())

        self.scale_button = customtkinter.CTkButton(master=self.scale_frame, text='Do Scaling', command=self.do_scaling)
        self.scale_button.grid(row=5, column=0, padx=20, pady=(20, 10))        
        
        # Preprocessing
        self.var_window_length = tk.IntVar(value=25)
        self.var_polyorder = tk.IntVar(value=2)
        self.var_deriv = tk.IntVar(value=1)

        self.checkbox_frame = customtkinter.CTkFrame(self, width=420)
        self.checkbox_frame.grid(row=1, column=2, columnspan=2, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.checkbox_frame, text='Choose Preprocessing', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.check_savgol = customtkinter.BooleanVar(value=False)
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_frame, text='Savgol_Filter', variable=self.check_savgol, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=12, weight='bold'))
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20)

        # self.dialog_savgol = customtkinter.CTkButton(master=self.checkbox_frame, text='Change Savgol Params', command=self.change_savgol_params)
        # self.dialog_savgol.grid(row=2, column=0, pady=(20, 0), padx=20)
        self.var_window_label = customtkinter.CTkLabel(self.checkbox_frame, text='Window_Length:', font=customtkinter.CTkFont(size=12))
        self.var_window_label.grid(row=2, column=0, padx=20, pady=(20,5), sticky='w')
        self.slider_winLength = customtkinter.CTkSlider(self.checkbox_frame, variable=self.var_window_length, from_=1, to=50, number_of_steps=49, command=self.update_window_length)
        self.slider_winLength.grid(row=3, column=0, padx=(20, 10), pady=(5, 5), sticky="e")
        self.update_window_length(self.slider_winLength.get())

        self.var_polyorder_label = customtkinter.CTkLabel(self.checkbox_frame, text='Polyorder:', font=customtkinter.CTkFont(size=12))
        self.var_polyorder_label.grid(row=4, column=0, padx=20, pady=(10,5), sticky='w')
        self.slider_polyorder = customtkinter.CTkSlider(self.checkbox_frame, variable=self.var_polyorder, from_=0, to=5, number_of_steps=5, command=self.update_polyorder)
        self.slider_polyorder.grid(row=5, column=0, padx=(20, 10), pady=(5, 5), sticky="e")
        self.update_polyorder(self.slider_polyorder.get())

        self.var_deriv_label = customtkinter.CTkLabel(self.checkbox_frame, text='Deriv:', font=customtkinter.CTkFont(size=12))
        self.var_deriv_label.grid(row=6, column=0, padx=20, pady=(10,5), sticky='w')
        self.slider_deriv = customtkinter.CTkSlider(self.checkbox_frame, variable=self.var_deriv, from_=0, to=5, number_of_steps=5, command=self.update_deriv)
        self.slider_deriv.grid(row=7, column=0, padx=(20, 10), pady=(5, 10), sticky="e")
        self.update_deriv(self.slider_deriv.get())

        self.check_snv = customtkinter.BooleanVar(value=False)
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_frame, text='SNV', variable=self.check_snv, onvalue=True, offvalue=False, font=customtkinter.CTkFont(size=12, weight='bold'))
        self.checkbox_2.grid(row=8, column=0, pady=(20, 10), padx=20)

        self.preprocessing_button = customtkinter.CTkButton(master=self.checkbox_frame, text='Do Preprocessing', command=self.do_preprocessing)
        self.preprocessing_button.grid(row=9, column=0, padx=20, pady=(20, 20))

        # Data splitting
        self.splitting_frame = customtkinter.CTkFrame(self, width=140)
        self.splitting_frame.grid(row=1, column=4, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Data Splitting', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Random State = 42:', font=customtkinter.CTkFont(size=12))
        self.preprocessing_label.grid(row=1, column=0, padx=(20, 20), pady=(20, 0))
        self.entry_randstate = customtkinter.CTkEntry(master=self.splitting_frame, placeholder_text='42')
        self.entry_randstate.grid(row=2, column=0, padx=(20, 20), pady=(10, 0), sticky="nsew")

        self.preprocessing_label = customtkinter.CTkLabel(self.splitting_frame, text='Test Size = 20 %:', font=customtkinter.CTkFont(size=12))
        self.preprocessing_label.grid(row=3, column=0, padx=(20, 20), pady=(20, 0))
        self.entry_split = customtkinter.CTkEntry(master=self.splitting_frame, placeholder_text='20')
        self.entry_split.grid(row=4, column=0, padx=(20, 20), pady=(10, 0), sticky="nsew")

        self.splitting_button = customtkinter.CTkButton(master=self.splitting_frame, text='Do Splitting', command=self.do_splitting)
        self.splitting_button.grid(row=5, column=0, padx=20, pady=(20, 10))
        
        # Regression
        self.regression_frame = customtkinter.CTkFrame(self, width=140)
        self.regression_frame.grid(row=1, column=5, sticky='nsew')
        self.preprocessing_label = customtkinter.CTkLabel(self.regression_frame, text='Regression', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.preprocessing_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.radio_var = tk.IntVar(value=0)
        # self.label_radio_group = customtkinter.CTkLabel(master=self.regression_frame, text="Method:")
        # self.label_radio_group.grid(row=1, column=0, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.regression_frame, text='PLS', variable=self.radio_var, value=0, font=customtkinter.CTkFont(size=12, weight='bold'))
        self.radio_button_1.grid(row=2, column=0, pady=(20,10), padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.regression_frame, text='SVM', variable=self.radio_var, value=1, font=customtkinter.CTkFont(size=12, weight='bold'))
        self.radio_button_2.grid(row=3, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.regression_frame, text='CNN', variable=self.radio_var, value=2, font=customtkinter.CTkFont(size=12, weight='bold'), state='disabled')
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
        self.text_frame.grid(row=1, column=6, columnspan=2, sticky='nsew')

        self.text_label = customtkinter.CTkLabel(self.text_frame, text='Console Output:', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.text_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.text_box = customtkinter.CTkTextbox(self.text_frame, height=400, width=280)
        self.text_box.grid(row=1, column=0, columnspan=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        sys.stdout = self

    def write(self, txt):
        # Append the text to the text box
        self.text_box.insert('end', txt)
        self.text_box.update_idletasks

    def flush(self):
        # This could be used to ensure the text box is updated promptly, but in this case it does nothing
        pass

    def update_window_length(self, value):
        self.label_window_length = customtkinter.CTkLabel(self.checkbox_frame, text=f'{value}')
        self.label_window_length.grid(row=3,column=1)    
    
    def update_polyorder(self, value):
        self.label_polyorder = customtkinter.CTkLabel(self.checkbox_frame, text=f'{value}')
        self.label_polyorder.grid(row=5,column=1, padx=(10,10)) 

    def update_deriv(self, value):
        self.label_deriv = customtkinter.CTkLabel(self.checkbox_frame, text=f'{value}')
        self.label_deriv.grid(row=7,column=1)

    def update_startWL(self, value):
        self.label_slider_startWL = customtkinter.CTkLabel(self.scale_frame, text=f'{value}')
        self.label_slider_startWL.grid(row=2,column=1)

    def update_stopWL(self, value):
        self.label_slider_stopWL = customtkinter.CTkLabel(self.scale_frame, text=f'{value}')
        self.label_slider_stopWL.grid(row=4,column=1)

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
        self.preset_startWL = int(self.var_startWL.get())
        self.preset_stopWL = int(self.var_stopWL.get())
        self.X_df_specific = self.X_df.loc[:, self.preset_startWL:self.preset_stopWL]
        self.X = self.X_df_specific.to_numpy()
        self.y = np.array(self.X_df_specific.index)
        self.n_wavelenths = self.X.shape[1]
        self.wl_int = np.linspace(self.preset_startWL, self.preset_stopWL, self.n_wavelenths)
        print('Number of used wavelengths: '+ str(self.n_wavelenths))
        plot_samples(self.wl_int, self.X, self.y)
        self.preprocessing_button.configure(state='enabled')
    
    # Preprocessing from Utils
    def do_preprocessing(self):
        savgol_wl = int(self.var_window_length.get())
        savgol_poly = int(self.var_polyorder.get())
        savgol_deriv = int(self.var_deriv.get())
        if self.check_savgol.get() == True and self.check_snv.get() == False:
            self.X = savgol(self.X, savgol_wl, savgol_poly, deriv=savgol_deriv)
            print('Savgol done')
        
        elif self.check_savgol.get() == False and self.check_snv.get() == True:
            self.X = snv(self.X)
            print('SNV done')
        
        elif self.check_savgol.get() == True and self.check_snv.get() == True:
            self.X = savgol(self.X, savgol_wl, savgol_poly, deriv=savgol_deriv)
            self.X = snv(self.X)
            print('Savgol AND SNV done')
        
        else:
            self.X = self.X
            print('No Preprocessing done')

        plt.figure(figsize=(9,5))
        plt.plot(self.wl_int, self.X[:20,:].T)
        plt.title('Preprocessed spectra')
        plt.show()

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, random_state=self.random_state)
        
        plt.figure(figsize=(9,5))
        plt.title('Distribution of Y train and test data')
        sns.histplot(self.y_train,label='train Y', kde=True, stat='density')
        sns.histplot(self.y_test,label='test Y', kde=True, stat='density')
        plt.legend()
        plt.show()
        
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
            pls.fit(self.X_train, self.y_train)

            y_c = pls.predict(self.X_train)
            y_cv = pls.predict(self.X_test)
            #y_vv = pls.predict(self.X_val)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            #score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            #rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print_metrics(score_c, score_cv, rmse_c, rmse_cv)
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_cv, rmse_c, rmse_cv, self.X_train, self.X_test, self.n_wavelenths, self.preset_startWL, self.preset_stopWL)

        # SVM
        elif self.radio_var.get() == 1:
            print('Starting SVM Regression')
            parametersSVM = {'C': [1, 100, 1000, 20000, 40000, 60000, 70000], \
                             'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 20]}
            set_kernel = 'rbf'
            svm = SVR(kernel=set_kernel)
            cvSVM = 10 #5
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
            svm.fit(self.X_train, self.y_train)

            y_c = svm.predict(self.X_train)
            y_cv = svm.predict(self.X_test)
            #y_vv = svm.predict(self.X_val)

            score_c = r2_score(self.y_train, y_c)
            score_cv = r2_score(self.y_test, y_cv)
            #score_vv = r2_score(self.y_val, y_vv)
            rmse_c = root_mean_squared_error(self.y_train, y_c)
            rmse_cv = root_mean_squared_error(self.y_test, y_cv)
            #rmse_vv = root_mean_squared_error(self.y_val, y_vv)

            print_metrics(score_c, score_cv, rmse_c, rmse_cv)
            plot_metrics(self.radio_var.get(), self.y_test, y_cv, score_c, score_cv, rmse_c, rmse_cv, self.X_train, self.X_test, self.n_wavelenths, self.preset_startWL, self.preset_stopWL)

            # filename = "SVM_WL930_1692_savgol15_3_3+snv_rand42_v13_model.pkl"
            # pickle.dump(svm, open(filename, "wb"))

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

        filename = "SVM_WL930_1692_savgol15_3_3+snv_rand42_v13_model.pkl"
        pickle.dump(svm, open(filename, "wb"))

    # Function RESET    
    def reset(self):
        self.X = None
        self.y = None
        self.X_df = None
        self.X_df_specific = None
        self.n_wavelenths = None
        self.preset_startWL = self.startWL
        self.preset_stopWL = self.stopWL
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