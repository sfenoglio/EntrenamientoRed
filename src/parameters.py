## @package parameters
#--------------------------------------------------------------------------------
#                                 PARAMETROS
#--------------------------------------------------------------------------------
num_epochs = 30
batch_size = 2
learning_rate = 0.001
momentum= 0.9 #solo para SGD

#arquitectura
nombreArq= "RedW" #sin .py

# optimizador
tipoOptim= "Adam" #pueden ser "Adam", "Adadelta" o "SGD"

# funcion de perdida
tipoLoss= "CrossEntropy" #pueden ser "CrossEntropy", "BCE", "BCEwLogits" o "MSE"

#%%   DATA
cant_train= 2#38           #cantidad de modulos de entrenamiento para cargar
val_split= 0.15         #porcentaje de imagenes de entrenamiento usadas para validar
nameData= "etapa3"            #extra que se agrega al nombre para guardar (util para no sobreescribir)

#%%   MODELO
pathAnterior= "../modelo/Adam_CrossEntropy_etapa2"    #path a modelo a cargar. Si esta vacio, entrena desde 0
nameModel= "etapa3"       #extra que se agrega al nombre para guardar (util para no sobreescribir)

#%% PATHS
DATA_PATH = "../data/"
MODEL_STORE_PATH = "../modelo/"+tipoOptim+"_"+tipoLoss+"_"+nameModel
OUT_PATH= "../salida/"+nameData
PREDICT_PATH= "../salida/"+"predicciones/"+nameData
REALES_PATH= "../inferencias/"+nameData

#%% SALIDA
verGraficas= False #indica si ver salidas intermedias online
cadaCuantas= 1 #cada cuantas epocas guardar graficas de entrenamiento
cadaCuantos= 300 #cada cuantos batches guardar predicciones intermedias
cadaCuantosVal= 60 #cada cuantos batches guardar predicciones intermedias en validacion
cadaCuantosTest= 20 #cada cuantos batches guardar predicciones intermedias en test
