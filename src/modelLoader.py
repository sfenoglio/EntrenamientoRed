#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##@package modelLoader
# Paquete que define las funciones necesarias para el cargado y guardado de un modelo entrenado.
# Además, define funciones para obtener funciones de pérdida y optimizadores.

"""
Created on Wed Oct  3 12:05:49 2018

@author: sebastian
"""

import torch
import resource

#%%
## Función que guarda el modelo en el directorio indicado con la función 'save()' de PyTorch.
# Los parámetros entrenados se guardan en un diccionario con la clave 'state_dict' y el estado del optimizador como 'optim_dict', a fin de poder reanudar el entrenamiento.
# Además, almacena la época en que se hace con la clave 'epoch'.
#@param epoch Entero que indica la época en que se guarda el modelo.
#@param model Modelo de PyTorch al cual se quieren almacenar los parámetros entrenados.
#@param optimizer Optimizador que se utilizó en el entrenamiento y se quiere almacenar el estado.
#@param is_best Booleano que indica si se está guardando el modelo con menor pérdida o no, para guardarlos diferenciadamente. En el primer caso, le añade "_best" al nombre y en el segundo "_final".
#@param path Directorio en que se almacena.
def saveModel(epoch, model, optimizer, is_best=False, path= "./model/"):
    print("Guardando estado anterior de modelo...")
    
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optim_dict' : optimizer.state_dict()}
    
    if(is_best):
        path= path + "_best"
    else:
        path= path + "_final"
    
    torch.save(state, path)
    
    print("Estado anterior de modelo guardado.")
    

#%% 
## Función que carga el modelo desde el directorio indicado con la función 'load()' de PyTorch.
# Con la función 'load_state_dict()' del modelo de PyTorch se reestablecen los parámetros entrenados del modelo.
# Si se quiere cargar el optimizador para continuar el entrenamiento, se hace con la función 'load_state_dict()' del optimizador de PyTorch.
#@param path Directorio del modelo a cargar.
#@param model Arquitectura de PyTorch del modelo a cargar.
#@param optimizer Optimizador del cual se quiere reestablecer el estado previo al guardado. Si es 'None', no se carga.
#@return Devuelve una tupla que contiene el número de época en que se cargó, el modelo con los parámetros cargados y el optimizador con el estado reestablecido.
def loadModel(path, model, optimizer= None):
    print("Cargando estado anterior de modelo...")
    
    state= torch.load(path)
    
    model.load_state_dict(state["state_dict"])
    
    epoch= state["epoch"]
    
    if(optimizer):
        optimizer.load_state_dict(state["optim_dict"])
    
    print("Estado anterior de modelo cargado. (Memoria: "+str((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024)+"MB)")
    return (epoch,model,optimizer)


#%%
## Función que devuelve el optimizador de PyTorch requerido.
# Pueden ser gradiente descendiente estocástico "SGD", "Adam" o "Adadelta".
# En el primer caso se debe pasar la tasa de aprendizaje 'learning_rate' y el momentum 'momentum'.
# En el segundo, sólo la tasa de aprendizaje.
#Mientras que en el último no necesita ninguno de los dos ya que son parámetros que aprende durante el entrenamiento.
#@param model Arquitectura de PyTorch que se optimizará.
#@param name Cadena de texto que indica el optimizador que se quiere.
#@param learning_rate Tasa de aprendizaje.
#@param momentum Momentum, sólo utilizado en "SGD".
#@return Optimizador de PyTorch requerido.
def getOptimizer(model, name, learning_rate= 0.001, momentum=0.9):
    if(name=="Adam"):
        return torch.optim.Adam(model.parameters(), lr= learning_rate)
    
    if(name=="Adadelta"):
        return torch.optim.Adadelta(model.parameters())
    
    if(name=="SGD"):
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    #por defecto
    return torch.optim.Adam(model.parameters(), lr= learning_rate)
    


#%%
## Función que devuelve la función de pérdida de PyTorch requerida.
# Pueden ser "BCE" (Binary Cross Entropy), "BCEwLogits" (Binary Cross Entropy with Logits), "CrossEntropy" y "MSE" (Mean Squared Error).
#@param name Cadena de texto que indica la función a devolver.
#@return Función de pérdida indicada de PyTorch.
def getLossFunc(name):
    if(name=="BCE"):
        return torch.nn.BCELoss()
    
    if(name=="BCEwLogits"):
        return torch.nn.BCEWithLogitsLoss()
    
    if(name=="CrossEntropy"):
        return torch.nn.CrossEntropyLoss()
#        return torch.nn.functional.cross_entropy
    
    if(name=="MSE"):
        return torch.nn.MSELoss()
    
    #por defecto
    return torch.nn.MSELoss()
