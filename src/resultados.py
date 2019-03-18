#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package resultados
# Archivo en el que se definen las clases y funciones utilizadas para guardar los resultados parciales y finales que se obtienen.
# Incluye el cálculo de las medidas de recall y Jaccard, así como también la graficación de ellos y la pérdida.
# Estas gráficas se realizan para el entrenamiento y para la validación mediante la librería 'matplotlib'.
# También incluye la muestra online de predicciones y el almacenamiento de ellas.

"""
Created on Tue Sep 25 10:35:19 2018

@author: sebastian
"""

import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 


#%% creating a colormap
colores= [[0,0,0]]
for i in range(2,-1,-1):
    for j in range(2,-1,-1):
        for k in range(2,-1,-1):
            colores.append(np.array([
                    1/i if i!=0 else 0,
                    1/j if j!=0 else 0,
                    1/k if k!=0 else 0]))
cmap= matplotlib.colors.ListedColormap(colores[0:24])


#%% GRAFICAS
class graficas():
    ## Constructor de la clase, encargado inicializar la figura con las subfiguras en las que se graficarán los resultados.
    # Si 'online' es falso, las subfiguras son 6. De izquierda a derecha corresponden a la pérdida, recall y Jaccard promedio por época. Las tres superiores son de entrenamiento y las tres inferiores de validación.
    # Si 'online' es verdadero, las subfiguras son 9 ya que se muestran también predicciones parciales de la red.
    #@param online Booleano que indica si las gráficas se muestran mientras se entrena o solamente se guardan en un archivo por época.
    #@param size Tamaño de la figura.
    def __init__(self, online= True, size= (15,8)):
        self.online= online
        self.contador= 0
        if(online):
            #incializo subplots
            plt.ion()
            self.fig= plt.figure(1, figsize=size)
            [self.sp1,self.sp2,self.sp3],[self.sp4,self.sp5,self.sp6],[self.sp7,self.sp8,self.sp9] = self.fig.subplots(3,3)
        else:
            plt.ioff()
            self.fig= plt.figure(1, figsize=size)
            [self.sp1,self.sp2,self.sp3],[self.sp4,self.sp5,self.sp6] = self.fig.subplots(2,3)
    
    #%% 
    ## Función que actualiza las predicciones en la figura si el entrenamiento es 'online', sino solamente las almacena como archivo.
    #@param epoch Número de época utilizado para guardar las predicciones.
    #@param maskPredicted1 Arreglo 2D de NumPy correspondiente a la imagen que se quiere mostrar y guardar. Generalmente es la predicción de la red.
    #@param maskPredicted2 Arreglo 2D de NumPy correspondiente a la imagen que se quiere mostrar y guardar. Generalmente es el ground truth o la imagen de la que se predijo.
    #@param path Directorio en que se guardan 'maskPredicted1' y 'maskPredicted2'.
    #@param name Cadena de texto extra que se agrega al nombre de la magen al guardarla.
    #@param real Booleano que si es verdadero guarda 'maskPredicted2' en escala de grises por corresponder a cromosomas, sino se guarda en colores por corresponder a una máscara que indica las clases.
    def cambiarPredicted(self,epoch,maskPredicted1,maskPredicted2,path="./",name="",real=False):
#        if(maskPredicted2.ndim==3):
#            #entonces es borde o lineal
#            maskPredicted1= maskPredicted1[0]
#            maskPredicted2= maskPredicted2[0]
#        else:
            #entonces es maskcanales, 
        maskPredicted1 = np.argmax(maskPredicted1, axis=0)
            #maskPredicted2 ya esta bien
            
                    
        if(self.online):        
            #prediccion
            if(type(maskPredicted1)==np.ndarray):
                self.sp8.cla()
                self.sp8.set_xlabel("Predicción")
                self.sp8.imshow(maskPredicted1, cmap=cmap, vmin= 0, vmax= 23)
        
            #prediccion validacion
            if(type(maskPredicted2)==np.ndarray):
                self.sp9.cla()
                self.sp9.set_xlabel("Ground truth")
                self.sp9.imshow(maskPredicted2, cmap=cmap, vmin= 0, vmax= 23)
                
            #pausa corta
            plt.pause(0.01)
    
        #siempre guardo prediccion
        fig2= plt.figure(2)
        [im1, im2]= fig2.subplots(1,2)
        im1.imshow(maskPredicted1, cmap=cmap, vmin= 0, vmax= 23)
        im1.set_xlabel("Predicción")
        if(real):
            im2.imshow(1-maskPredicted2, cmap="gray")
            im2.set_xlabel("Imagen sobre la que se predijo")
        else: 
            im2.imshow(maskPredicted2, cmap=cmap, vmin= 0, vmax= 23)
            im2.set_xlabel("Ground truth")
        self.contador += 1
        fig2.savefig(path+"ep"+str(epoch)+"_"+name+str(self.contador)+".png")
        plt.close(2)


#%%
    ## Función que actualiza las gráficas de pérdida, recall y Jaccard de entrenamiento y de validación.
    # Además, si se indica, se guardan las medidas obtenidas en un archivo '.npz' con claves iguales a los nombres de los parámetros.
    #@param epoch Número de época utilizado para guardar las predicciones.
    #@param loss Lista que contiene la pérdida de entrenamiento en cada época.
    #@param acc Lista que contiene el recall de entrenamiento en cada época.
    #@param jacc Lista que contiene el Jaccard de entrenamiento en cada época.
    #@param loss_val Lista que contiene la pérdida de validación en cada época.
    #@param acc_val Lista que contiene el recall de validación en cada época.
    #@param jacc_val Lista que contiene el Jaccard de validación en cada época.
    #@param guardar Booleano que indica si se guardan las medidas en un archivo '.npz' o no.
    #@param ultimo Booleano que indica si es la última gráfica, entonces espera que se aprete un botón para cerrarse.
    #@param name Cadena de texto extra que se agrega al nombre de la magen al guardarla.
    def graficar(self,epoch,loss,acc,jacc,loss_val,acc_val,jacc_val,path="./",guardar=True,ultimo=False,name=""):
                    
        #%% train
        #loss
        self.sp1.cla()
        self.sp1.set_title("training loss")
        self.sp1.grid(True)
        self.sp1.plot(loss)
        
        #acc
        self.sp2.cla()
        self.sp2.set_title("training recall")
        self.sp2.grid(True)
        self.sp2.plot(acc)
        
        #jaccard
        self.sp3.cla()
        self.sp3.set_title("training jaccard")
        self.sp3.grid(True)
        self.sp3.plot(jacc)
        
        #%%validation
        #loss
        self.sp4.cla()
        self.sp4.set_title("validation loss")
        self.sp4.grid(True)
        self.sp4.plot(loss_val)
        
        #acc
        self.sp5.cla()
        self.sp5.set_title("validation recall")
        self.sp5.grid(True)
        self.sp5.plot(acc_val)
        
        #jaccard
        self.sp6.cla()
        self.sp6.set_title("validation jaccard")
        self.sp6.grid(True)
        self.sp6.plot(jacc_val)
        
        #%% extra    
        #si pide guardo
        self.fig.savefig(path+"_final.png")
        if(not self.online):
            plt.close(self.fig)
        else:
            #pausa corta para poder ver
            plt.pause(0.01)
        
        #medidas
        if(guardar):
            #metricas por las dudas
            np.savez_compressed(path+"_metricsTrain.npz",
                                loss_train= loss,
                                acc_train= acc,
                                jacc_train= jacc,
                                loss_val= loss_val,
                                acc_val= acc_val,
                                jacc_val= jacc_val)
        
        
        #si es el ultimo espero boton
        if(ultimo):
            plt.waitforbuttonpress()



#%% LOGGING
logger = logging.getLogger('log')
## Función que inicializa el logger utilizado para guardar cada paso en el entrenamiento.
# Utiliza la librería 'logging' que trae por defecto Python.
#@param filename Nombre del archivo en el que se guarda.
def inicializarLogger(filename='log.txt'):
    hdlr = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

## Función que guarda en el archivo de log la cadena que se le pasa.
#@param cadena Cadena de texto que se guarda en el archivo de log.
#@param porConsola Booleano que indica si la cadena también se muestra por consola.
def log(cadena,porConsola=True):
    logger.info(cadena)
    if(porConsola):
        print(cadena)

#%% devuelve acc, jaccard 
## Función que calcula el recall y el Jaccard promedio de un batch de imágenes dados sus correspondientes ground truths.
# Para ello, primero calcula la clase de cada píxel mediante la función 'argmax' de NumPy.
# A continuación, calcula la matriz de confusión mediante la función 'confusion_matrix'.
# Luego, recorre cada fila (correspondiente a cada clase) y, si hay algo en ella, hace el cálculo de las medidas y las añade a una lista.
# Si son todos ceros significa que esa clase existe pero nunca fue acertada y traería problemas de división por cero.
# En ese caso, directamente se añade un recall y Jaccard de valor 0.
# Se calcula el recall como el valor de la diagonal dividido la suma de la fila [TP / (TP+FN)].
# El Jaccard es el valor de la diagonal dividido la suma de la fila y de la columna [TP / (TP+FN+FP)].
# Entre corchetes se indica la fórmula, significando TP verdaderos positivos, FN falsos negativos y FP falsos positivos.
# Por último, se promedia el recall y el Jaccard y se devuelve como tupla.
#@param predicted Arreglo de cuatro dimensiones de NumPy. La primera corresponde a la cantidad de datos, la segunda a la cantidad de canales (clases) y las restantes al tamaño de la imagen.
#@param labels Arreglo de NumPy de 3 dimensiones que contiene el número de clase de cada píxel. La primera dimensión es la cantidad de datos y las restantes el tamaño de la imagen. El número indica la clase correcta de cada clase o equivalentemente el canal de 'predicted' correcto.
#@return Tupla conteniendo los valores promedio de recall y Jaccard.
def medidas(predicted, labels):
    #necesito calcular el maximo para todos => CON LOS INDICES YA TENGO LOS LABELS
    predicted = np.argmax(predicted, axis=1)
    
    #confusion matrix
    cm= confusion_matrix(labels.ravel(), predicted.ravel())
    
    acc_clase= []
    jacc_clase= []
    for i in range(1,cm.shape[0]):
        #verifico que haya algo en la fila: si no hay FN ni TP es que no estaba esa clase en y_true
        filaSuma= cm[i,:].sum()
        if(filaSuma>0):
            acc_clase.append( cm[i,i]/filaSuma ) #TP / (TP+FN)
            jacc_clase.append(cm[i,i] / (filaSuma + cm[:,i].sum() - cm[i,i])) #TP/(TP+FN+FP) #menos TP para que no lo cuente dos veces
        else:
            acc_clase.append( 0 ) 
            jacc_clase.append( 0 ) 
    
    acc= np.mean(acc_clase)
    jacc= np.mean(jacc_clase)
        
    return (acc,jacc)