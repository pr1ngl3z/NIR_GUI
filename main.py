import os
import numpy as np
import tkinter as tk
import customtkinter
import sys

from preprocessing import msc, snv, savgol
from sklearn.model_selection import train_test_split

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

        # Create a text box
        self.text_frame = customtkinter.CTkFrame(self, width=280)
        self.text_frame.grid(row=1, column=3, sticky='nsew')
        self.text_label = customtkinter.CTkLabel(self.text_frame, text='Console Output:', font=customtkinter.CTkFont(size=14, weight='bold'))
        self.text_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.text_box = customtkinter.CTkTextbox(self.text_frame)
        self.text_box.grid(row=1, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Redirect stdout to the text box
        sys.stdout = self

    def write(self, txt):
        # Append the text to the text box
        self.text_box.insert('end', txt)

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
            self.random_state = self.entry_randstate.get()
        else:
            self.random_state = 42
        print(self.random_state)

        if len(self.entry_split.get()) > 0:
            self.entry_split_answer = float(self.entry_split.get())
            self.test_size = round((self.entry_split_answer/100.0),2)
        else:
            self.test_size = 0.2
        print(self.test_size)

        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, random_state=self.random_state)
        print('SUCCESS Data splitting')


if __name__ == "__main__":
    app = App()
    app.mainloop()