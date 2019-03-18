
## @package inferir_parameters

#--------------------------------------------------------------------------------
#                                 PARAMETROS
#--------------------------------------------------------------------------------
batch_size= 10

#arquitectura
nombreArq= "RedW" #sin .py

#%%   DATA
nombreData= "reales"      #para leer, sin .npz
hayLabels= False 	# cuando no necesito labels
cant_inferir= 1           #cantidad de modulos de inferir para cargar (solo cuando uso otros de train)
nameData= "reales"            #para escribir, extra que se agrega al nombre para guardar (util para no sobreescribir)

#%%   MODELO
pathModel= "../modelo/Adam_CrossEntropy_etapa2"               #path a modelo a cargar

# loss. Solo utilizada si hay labels
#tipoLoss= "BCE"
#tipoLoss= "BCEwLogits"
tipoLoss= "CrossEntropy"
#tipoLoss= "MSE"

#%%paths
DATA_PATH = "../data/"
PREDICT_PATH= "../inferencias/"+nameData+"Predicciones/"
OUT_PATH= "../inferencias/"+nameData

#%% graficas
verGraficas= False #graficas online o no
cadaCuantas= 1 #cada cuantas imágenes guardar mascara predicha
