#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package inferir
# Script que itera sobre los datos para inferir con la red convolucional sobre ellos.
# Es la versión simplificada de 'trainModel.py'.

"""
Created on Thu Oct  4 20:06:31 2018

@author: sebastian
"""

import torch
from torch import nn
import dataLoader
import modelLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import resultados
import cv2 as cv

#%%
#--------------------------------------------------------------------------------
#                               INICIALIZACIONES
#--------------------------------------------------------------------------------
#parametros
from inferir_parameters import *

#imagenes
graficas= resultados.graficas(online=verGraficas)

#LOG
resultados.inicializarLogger(OUT_PATH+'log.txt')
f= open("inferir_parameters.py")
resultados.log(f.read())
f.close()


#%%
#--------------------------------------------------------------------------------
#                               MODELO
#--------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# cargo modelo
#model= OverlapSegmentationNet.OverlapSegmentationNet().cuda()
exec("import "+nombreArq)
exec("model= "+nombreArq+"."+nombreArq+"().to(device)")

# loss
criterion= modelLoader.getLossFunc(tipoLoss)
resultados.log(criterion.__str__())

#cargar anterior
_, model, _= modelLoader.loadModel(pathModel, model, None)


#%%
#--------------------------------------------------------------------------------
#                                   DATA
#--------------------------------------------------------------------------------

inf_dataset = dataLoader.genDataset(nombreData, hayLabels, cant_inferir, path= DATA_PATH)
inf_loader= dataLoader.splitAndLoader(inf_dataset, val_split= 0, batch_size=batch_size)

#creacion de carpetas
try:
    os.stat(PREDICT_PATH)
except:
    os.mkdir(PREDICT_PATH)


#%%
#--------------------------------------------------------------------------------
#                                 INFERENCE
#--------------------------------------------------------------------------------

contador= 0
total_step = len(inf_loader)
model.eval()
with torch.no_grad():
    #segun haya o no labels
    #%% NO HAY LABELS
    if(not hayLabels):
        for i, (images) in enumerate(inf_loader):
            #gpu
            images = images.cuda()
            
            #forward
            outputs= model(images)
            
            #track
            resultados.log('Step [{}/{}]'.format(i+1, total_step))
            
            #guardo todo
            predicted = (outputs.data.cpu()).numpy()
            for j in range(predicted.shape[0]):
                contador+= 1
                if(contador%cadaCuantas==0):
                    maskPredicted= np.array(predicted[j])
                    maskLabel= np.array(images[j,0].cpu())
                    graficas.cambiarPredicted(0, maskPredicted, maskLabel, name= "inf", path=PREDICT_PATH, real=True)
                    cv.imwrite(PREDICT_PATH+str(contador)+"_real.png", 255-(255*maskLabel).astype(np.uint8))
                    
                    
    #%% SI HAY LABELS
    else:
        loss_inf= []
        acc_inf= []
        jacc_inf= []
        for i, (images, labels) in enumerate(inf_loader):
            #GPU
            images, labels = images.cuda(), labels.cuda()
            
            #forward
            outputs= model(images)
            loss = criterion(outputs, labels)
            loss_inf.append(loss.item())
            
            #%% Track the accuracy and jaccard
            predicted = (outputs.data.cpu()).numpy()        
            acc, jacc = resultados.medidas(predicted, (labels.data.cpu()).numpy(), tol)
            acc_inf.append(acc)
            jacc_inf.append(jacc)
            
            #%%        
            #logger
            resultados.log('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}, Jaccard: {:.3f} %'
                           .format(i+1, total_step, loss.item(), acc * 100, jacc * 100), porConsola= True)
    
            #guardar
            for j in range(predicted.shape[0]):
                contador+= 1
                if(contador%cadaCuantas==0):
                    maskPredicted= np.array(predicted[j])
                    maskLabel= np.array(labels[j])
                    graficas.cambiarPredicted(0, maskPredicted, maskLabel, name= "inf", path=PREDICT_PATH)
        
        #metricas
        np.savez_compressed(OUT_PATH+"_metricsInf.npz",
                                loss_inf= loss_inf,
                                acc_inf= acc_inf,
                                jacc_inf= jacc_inf)
        
        #final log
        resultados.log('\n FINAL, Loss: {:.3f}, Accuracy: {:.3f}, Jaccard: {:.3f} %'
                   .format(np.mean(loss_inf), np.mean(acc_inf) * 100, np.mean(jacc_inf) * 100), porConsola= True)

