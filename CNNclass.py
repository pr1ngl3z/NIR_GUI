import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal # type: ignore
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # Hypervariablen 
        self.INPUT_DIMS = np.shape(self.X_train)[1]
        self.CONV1D_DIMS = self.INPUT_DIMS
        self.K_NUMBER = 2 #1
        self.K_WIDTH = 5
        self.K_STRIDE = 1
        self.FC1_DIMS = 36
        self.FC2_DIMS = 18
        self.FC3_DIMS = 12
        self.OUT_DIMS = 1
        self.DROPOUT = 0.05
        self.EPOCHS = 10000
        self.BATCH = 1024
        self.LR_Adam = 0.005*self.BATCH/256. #0.01
        self.LR_RMS = 0.0005*self.BATCH/256. 

        # Variablen L2 Regularisierung
        self.beta = 0.003/2.
        self.K_REG = tf.keras.regularizers.l2(self.beta)

        # Initialisierung
        self.K_INIT = HeNormal(seed=42)

        self.reproducible_comp()
        self.model = self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential([
            layers.Reshape((self.INPUT_DIMS, 1), input_shape=(self.INPUT_DIMS,)),
            layers.Conv1D(filters=self.K_NUMBER,
                        kernel_size=self.K_WIDTH,
                        strides=self.K_STRIDE,
                        padding='same', 
                        kernel_initializer=self.K_INIT,
                        kernel_regularizer=self.K_REG,
                        activation='elu',
                        input_shape=(self.CONV1D_DIMS,1)),
            layers.Dropout(self.DROPOUT),
            layers.Flatten(),
            layers.Dense(self.FC1_DIMS,
                        kernel_initializer=self.K_INIT,
                        kernel_regularizer=self.K_REG,
                        activation='elu'),
            layers.Dropout(self.DROPOUT),
            layers.Dense(self.FC2_DIMS,
                        kernel_initializer=self.K_INIT,
                        kernel_regularizer=self.K_REG,
                        activation='elu'),
            layers.Dropout(self.DROPOUT),
            layers.Dense(self.FC3_DIMS,
                        kernel_initializer=self.K_INIT,
                        kernel_regularizer=self.K_REG,
                        activation='elu'),
            layers.Dropout(self.DROPOUT),
            layers.Dense(1,
                        kernel_initializer=self.K_INIT,
                        kernel_regularizer=self.K_REG,
                        activation='linear')
            ])
        return self.model

    def compile_model(self, model):
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.LR_RMS), loss='mse', metrics=['mse'])

    def fit_model(self, X_train_scaled_rowcol, y_train, X_val_scaled_rowcol, y_val, callback):
        fitted_model=self.model.fit(X_train_scaled_rowcol, y_train, batch_size=self.BATCH, epochs=self.EPOCHS, 
             validation_data=(X_val_scaled_rowcol, y_val),
             verbose=2,
             callbacks=callback)
        return fitted_model


    def plot_loss(self, h1):
        with plt.style.context('ggplot'):
            plt.plot(h1.history['loss'], label='Training loss')
            plt.plot(h1.history['val_loss'], label='Validation loss')
            plt.yscale('log')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend()
            plt.show()
    
    # def standardize_row(self, X_train, X_val, X_test):
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train.T)
    #     X_val_scaled = scaler.fit_transform(X_val.T)
    #     X_test_scaled = scaler.fit_transform(X_test.T)
    #     return [X_train_scaled.T, X_val_scaled.T, X_test_scaled.T]


    @staticmethod
    def standardize_row(X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    @staticmethod
    def standardize_column(X_train, X_val, X_test):
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    
    def reproducible_comp(self):
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)