#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package dataLoader
# Paquete que define las clases y funciones necesarias para el cargado de los datos.
# Define la clase 'genDataset', la cual hereda de la clase 'Dataset' de PyTorch, a fin de cargar en él los datos.
# La misma almacena los datos en memoria con formato NumPy y sólo transforma a Tensor de PyTorch los que está usando en el momento para pasar la red.
# Luego, define la función 'splitAndLoader()' que toma el dataset generado para crear un 'DataLoader' de PyTorch, el cual provee facilidades para iterar por batch sobre el dataset.
# Además, la función anterior reserva una partición de los datos para validación, si asi se indicase.

"""
Created on Tue Sep 18 19:57:13 2018

@author: sebastian
"""

import torch
import torch.utils.data as utils
#from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchvision.transforms as trans
import resource


#%% paraQue= "train" o "test"
# cant_train= cantidad de modulos de entrenamiento a leer
class genDataset(utils.Dataset):
    
    ## Constructor de la clase que lee los datos del disco.
    # Si los datos fueran para entrenar (para "train") carga en 'data' todos los archivos con forma "trainN.npz", donde N es un número del 1 a 'cant_train'.
    # De lo contrario, simplemente carga el archivo indicado.
    # También se le pasa como parámetro si los datos tienen ground truth o no, para que los cargue o no en 'labels'.
    # Además, define la transformación que se le realizarán a los datos para normalizarlos con media 0.5 y desvío 0.5 mediante la librería 'torchvision'.
    #@param paraQue Cadena que indica si es para "train" u en otro caso es el nombre del archivo (sin '.npz').
    #@param hayLabels Booleano que indica si los datos tienen ground truth o no.
    #@param cant_train Entero que indica la cantidad de archivos de entrenamiento que hay para el caso de "train".
    #@param path Carpeta en la que se encuentran los mencionados archivos.
    def __init__(self, paraQue, hayLabels=True, cant_train=1, path= "./data/"):
        print("Cargando datos...")
        
        self.hayLabels= hayLabels
        
        #transformacion
        self.transform= trans.Normalize(mean=[0.5],
                                 std=[0.5])
        
        #%%
        #1. Leo segun paraQue, si es para entrenar concateno
        if(paraQue=="train"):
            #1.1 inicializo
            loaded= np.load(path+paraQue+str(1)+".npz")
            self.data= loaded["data"]
            if(self.hayLabels):
                self.labels= loaded["maskLineal"]
                
            
            #1.2 recorro segun la cantidad de modulos que pide
            for i in range(1,cant_train):
                loaded= np.load(path+paraQue+str(i+1)+".npz")
                
                #1.2 concateno
                self.data= np.concatenate((self.data, loaded["data"]))
                if(self.hayLabels):
                    self.labels= np.concatenate((self.labels, loaded["maskLineal"] ))
        
        
        #%%
        #2. sino directamente cargo
        else:
            #2.1 si es test es facil
            loaded= np.load(path+paraQue+".npz")
            #2.2 guardo datos como numpy. Inversa de los datos
            self.data= loaded["data"]
            if(self.hayLabels):						
                self.labels= loaded["maskLineal"]

        print("Datos cargados. (Memoria: "+str((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024)+"MB)")



#%%
    ## Sobrecarga del método homólogo de la clase 'Dataset' de Pytorch.
    # Define cómo se obtiene una imagen y su ground truth (si lo hubiera) del dataset.
    # Además, los transforma a Tensor de PyTorch y aplica la transformación de normalización definida en '__init__()'.
    #@param index Índice del dato a obtener.
    #@return Imagen y ground truth (si lo hubiera) como Tensor de PyTorch de 3 y 4 dimensiones respectivamente.
    def __getitem__(self, index):
        """
        :param index:
        :return: tuple (img, target) with the input data and its label
        """
        
        # load image and apply transforms
        img = (self.data[index]/255).astype(np.float32)
        img = torch.from_numpy(img)

        # apply transforms
        img = self.transform(img)
        
        # load labels
        if(self.hayLabels):
            target = (self.labels[index]).astype(np.long)
            target = torch.from_numpy(target)
        else:
            #si es vacio no devuelve target
            return img

        return img, target
    
    
    #%%
    ## Sobrecarga del método homólogo de la clase 'Dataset' de Pytorch.
    # Define cómo se obtiene la longitud del dataset.
    def __len__(self):
        return self.data.shape[0]

        
#%% si val_split==0, simplemente hace un loader
## Función que genera una partición de validación y transforma el dataset en un 'DataLoader' de Pytorch.
# Si el porcentaje de los datos a reservar para validación 'val_split' es mayor a 0, se generan dos DataLoaders.
# Uno para validación con el porcentaje indicado y otro para entrenar con lo restante.
# Esta división se realiza mezclando los índices de los datos con la función 'shuffle' de NumPy, a la cual se le puede proporcionar una semilla 'random_seed' para poder hacer reproducible el experimento.
# Luego, con la función 'SubsetRandomSampler()' de PyTorch se generan las dos particiones para crear los DataLoaders.
# A éstos últimos también debe pasársele el tamaño de batch que se utilizará.
#@param dataset Dataset de la clase definida anteriormente al cual se procesará.
#@param val_split Porcentaje de datos reservados para validación. Si fuese 0, no se reserva ninguno y se devuelve sólo un DataLoader con todos los datos.
#@param batch_size Tamaño de batch que usará el DataLoader.
#@param random_seed Semilla que se le pasa a NumPy para hacer reproducible el experimento.
#@return Una tupla que contiene el DataLoader de entrenamiento y el DataLoader de validación, o simplemente un único DataLoader con todos los datos si 'val_split' es 0.
def splitAndLoader(dataset, val_split, batch_size, random_seed= 42):
    #dividirlo
    #%%
    if(val_split>0):
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
    
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                   sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=valid_sampler)
    
        return train_loader, val_loader
    
    #%% sino simplente hago un loader
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size )