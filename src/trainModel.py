#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package trainModel
# Script que itera sobre los datos de entrenamiento para entrenar la red convolucional.

"""
Created on Tue Sep 18 21:09:55 2018

@author: sebastian
"""

#%%
import torch
from torch import nn
import dataLoader
import modelLoader
import numpy as np
import resultados

#%%
#--------------------------------------------------------------------------------
#                               INICIALIZACIONES
#--------------------------------------------------------------------------------
#parametros
from parameters import *

#logger
resultados.inicializarLogger(OUT_PATH+'log.txt')
f= open("parameters.py")
resultados.log(f.read())
f.close()

#salidas
graficas= resultados.graficas(online=verGraficas)

#mejor modelo
best_loss= -1


#%%
#--------------------------------------------------------------------------------
#                               MODELO
#--------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# cargo modelo
#model= OverlapSegmentationNet.OverlapSegmentationNet().to(device)
exec("import "+nombreArq)
exec("model= "+nombreArq+"."+nombreArq+"().to(device)")

# loss
criterion= modelLoader.getLossFunc(tipoLoss)
resultados.log(criterion.__str__())

#optimizer
optimizer = modelLoader.getOptimizer(model, tipoOptim, learning_rate, momentum)
resultados.log(optimizer.__str__())

#ver si hay que cargar anterior
if(pathAnterior!=""):
    nextEpoch, model, optimizer= modelLoader.loadModel(pathAnterior, model, optimizer)
else:
    nextEpoch= 0


#%%
#--------------------------------------------------------------------------------
#                                   DATA
#--------------------------------------------------------------------------------
#train_loader= dataLoader.getDataLoader("train", cant_train, batch_size=batch_size)
full_dataset = dataLoader.genDataset("train", True, cant_train, path= DATA_PATH)

train_loader, val_loader= dataLoader.splitAndLoader(full_dataset, val_split, batch_size)
del full_dataset

#reales loader
reales_dataset = dataLoader.genDataset("reales", False, 0, path= DATA_PATH)
reales_loader= dataLoader.splitAndLoader(reales_dataset, val_split= 0, batch_size=batch_size)


#%%
#--------------------------------------------------------------------------------
#                                   TRAINING
#--------------------------------------------------------------------------------

# inicializaciones
total_step = len(train_loader)
total_step_val = len(val_loader)
loss_list = []
acc_list = []
jacc_list = []
loss_list_val = []
acc_list_val = []
jacc_list_val = []

num_epochs += nextEpoch
for epoch in range(nextEpoch, num_epochs):
    loss_epoch= []
    acc_epoch= []
    jacc_epoch= []
    
    #%% para cada batch
    for i, (images, labels) in enumerate(train_loader):
        #GPU #.type_as(torch.cuda.HalfTensor())
        images, labels= images.to(device), labels.to(device)
        
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_epoch.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #%% Track the accuracy and jaccard
        predicted = (outputs.data.cpu()).numpy()
        acc, jacc = resultados.medidas(predicted, (labels.data.cpu()).numpy())
        acc_epoch.append(acc)
        jacc_epoch.append(jacc)
        
        #logger
        resultados.log('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}%, Jaccard: {:.3f}%'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                      acc * 100, jacc * 100), porConsola= True)
        
        
        #%% Guardar prediccion por batch. Siempre guarda pero se ve a veces        
        if((i+1)%cadaCuantos==0):
            maskPredicted= np.array(predicted[0])
            maskLabel= np.array(labels[0])
            graficas.cambiarPredicted(epoch, maskPredicted, maskLabel, path=PREDICT_PATH)
    
    

    #%%
    #--------------------------------------------------------------------------------
    #                                   VALIDACION
    #--------------------------------------------------------------------------------
    #para cada dato calcular la salida sin entrenar  (torch.no_grad())
    loss_val= []
    acc_val= []
    jacc_val= []
    with torch.no_grad():
        #%% para cada batch
        for i, (images, labels) in enumerate(val_loader):
            #GPU #.type_as(torch.cuda.HalfTensor())
            images, labels= images.to(device), labels.to(device)
            
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_val.append(loss.item())
            
            #%% Track the accuracy and jaccard
            predicted = (outputs.data.cpu()).numpy()
            acc, jacc = resultados.medidas(predicted, (labels.data.cpu()).numpy())
            acc_val.append(acc)
            jacc_val.append(jacc)
            
            #logger
            resultados.log('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}%, Jaccard: {:.3f}%'
              .format(epoch + 1, num_epochs, i + 1, total_step_val, loss.item(),
                      acc * 100, jacc * 100), porConsola= True)
    
            #%% Guardar prediccion por batch. Siempre guarda pero se ve a veces            
            if((i+1)%cadaCuantosVal==0):
                maskPredicted= np.array(predicted[0])
                maskLabel= np.array(labels[0])
                graficas.cambiarPredicted(epoch, maskPredicted, maskLabel, name= "val", path=PREDICT_PATH)
                
                
                
    #%%
    #--------------------------------------------------------------------------------
    #                                   REALES
    #--------------------------------------------------------------------------------
        #pruebo con cromosomas reales
        for i, (images) in enumerate(reales_loader):
            #gpu
            images = images.to(device)
            
            #forward
            outputs= model(images)
            
            #guardo
            predicted = (outputs.data.cpu()).numpy()
            for j in range(0,predicted.shape[0],4):
                maskPredicted= np.array(predicted[j])
                maskLabel= (images[j,0].cpu()).numpy()
                graficas.cambiarPredicted(epoch, maskPredicted, maskLabel, name= "real", path=REALES_PATH)




    #%%
    #--------------------------------------------------------------------------------
    #                                 VISUAL POR EPOCA
    #--------------------------------------------------------------------------------
    loss_list.append(np.mean(loss_epoch))
#    loss_list.append(np.sum(loss_epoch)) #contempla mas la dispersion
    acc_list.append(np.mean(acc_epoch))
    jacc_list.append(np.mean(jacc_epoch))
    loss_list_val.append(np.mean(loss_val))
#    loss_list.append(np.sum(loss_epoch)) #contempla mas la dispersion
    acc_list_val.append(np.mean(acc_val))
    jacc_list_val.append(np.mean(jacc_val))
    
    
    if((epoch+1)%cadaCuantas==0):
        graficas.graficar(epoch,loss_list,acc_list,jacc_list,       #salidas por epoca
                            loss_list_val,acc_list_val,jacc_list_val, #salidas de validacion
                            path=OUT_PATH)          #extra
        
    #logger
    resultados.log('\n Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.3f}%, Jaccard: {:.3f}% \n'
                   .format(epoch + 1, num_epochs, loss_list[len(loss_list)-1], 
                    acc_list[len(acc_list)-1] * 100, jacc_list[len(jacc_list)-1] * 100), porConsola= True)
    resultados.log('Epoch [{}/{}], Loss Val: {:.4f}, Accuracy Val: {:.3f}%, Jaccard Val: {:.3f}% \n'
                   .format(epoch + 1, num_epochs, loss_list_val[len(loss_list_val)-1], 
                    acc_list_val[len(acc_list_val)-1] * 100, jacc_list_val[len(jacc_list_val)-1] * 100), porConsola= True)
    
    #%% MEJOR MODELO
    if(best_loss==-1 or loss_list_val[len(loss_list_val)-1]<best_loss):
        # Save the best model
        print("Guardando mejor modelo...")
        modelLoader.saveModel(epoch, model, optimizer, is_best=True, path= MODEL_STORE_PATH)
        print("Mejor modelo guardado...")
        
        #actualizo best_loss
        best_loss= loss_list_val[len(loss_list_val)-1]
    
   
    
#%%guardar cosas cuando sale
graficas.graficar(epoch,loss_list,acc_list,jacc_list,
                  loss_list_val,acc_list_val,jacc_list_val, #salidas de validacion
                  ultimo=False, path=OUT_PATH)

# Save the model final
print("Guardando modelo final...")
modelLoader.saveModel(epoch, model, optimizer, is_best=False, path= MODEL_STORE_PATH)
print("Modelo final guardado.")


#%%
#--------------------------------------------------------------------------------
#                                   TEST
#--------------------------------------------------------------------------------
#borrar anteriores
del train_loader
del val_loader

#datos test
test_dataset = dataLoader.genDataset("test", True, 0, path= DATA_PATH)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=False,
                                          num_workers=4)

#logger
resultados.log('\n -----------------------------TEST-----------------------------', porConsola= True)



#%%TEST
#inicializar
total_step_test = len(test_loader)
model.eval()
acc_test= []
jacc_test= []
loss_test= []
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        #GPU
        images, labels = images.to(device), labels.to(device)
        
        #forward
        outputs= model(images)
        loss = criterion(outputs, labels)
        loss_val.append(loss.item())
        
        #%% Track the accuracy and jaccard
        predicted = (outputs.data.cpu()).numpy()        
        acc, jacc = resultados.medidas(predicted, (labels.data.cpu()).numpy())
        acc_test.append(acc)
        jacc_test.append(jacc)
        
        #%%        
        #logger
        resultados.log('Step [{}/{}], Test Accuracy: {:.3f}, Jaccard: {:.3f} %'
                       .format(i+1, total_step_test, acc * 100, jacc * 100), porConsola= True)

        #graficas
        if((i+1)%cadaCuantosTest==0):
            maskPredicted= np.array(predicted[0])
            maskLabel= np.array(labels[0])
            graficas.cambiarPredicted(epoch, maskPredicted, maskLabel, name= "test", path=PREDICT_PATH)
        
#metricas
np.savez_compressed(OUT_PATH+"metricsTest.npz",
                        loss_test= loss_test,
                        acc_test= acc_test,
                        jacc_test= jacc_test)

resultados.log('\n TEST FINAL, Accuracy: {:.3f}, Jaccard: {:.3f} %'
               .format(np.mean(acc_test) * 100, np.mean(jacc_test) * 100), porConsola= True)

