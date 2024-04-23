import os
import numpy as np
import tkinter as tk
import customtkinter

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
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1100}x{580}")

        folder_path = customtkinter.filedialog.askdirectory()
        print(folder_path)


if __name__ == "__main__":
    app = App()
    app.mainloop()