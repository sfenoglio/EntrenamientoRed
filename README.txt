%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
--------     CONTENTS     --------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

0- REQUIREMENTS
1- INSTALLATION AND DIRECTORIES
2- SETTINGS
3- DATASET
4- OUTPUT FILES
5- EXAMPLES


=================================
0- REQUIREMENTS
=================================
- Python (Tested on Python 3.6)

- Python librarys:
  + logging
  + matplotlib (2.2.3)
  + numpy (1.15.2)
  + opencv-contrib-python (3.4.2.17)
  + opencv-python (3.4.2.17)
  + os
  + resource
  + scikit-learn (0.20.0)
  + tensorflow (1.10.1)
  + torch (0.4.1)
  + torchvision (0.2.1)

** This tool was tested with these libraries versions. However, older versions
   could also work.


=================================
1- INSTALLATION AND DIRECTORIES
=================================
Installation only requires to unzip the "EntrenamientoRed.zip" file and download
the trained model with the link provided in modelo/link.txt (because it's larger
than maximum size allowed by github).
This tool can be launched by typing in the command line placed in src folder:

>> python3 trainModel.py

or

>> python3 inferir.py


Directories:
  data:              images generated with GenData.
  inferencias:       output files of 'inferir'.
  modelo:            trained models.
  salida:            output files of training.
  src:               libraries and functions.


====================================
2- SETTINGS
====================================
The arguments of training are stored in "src/parameters.py".

--- main parameters ---

 num_epochs                ---> limit of epochs to training
 batch_size                ---> batch size
 learning_rate             ---> learning rate
 nombreArq                 ---> path to the file that defines the architecture
 tipoOptim                 ---> name of optimizer to use
 tipoLoss                  ---> name of loss function to use
 cant_train                ---> number of training data files to load
 val_split                 ---> percentage of training data to save for validation
 pathAnterior              ---> path to the file that contains the trained parameters


All the parameters like data path, output path, model saving path, etc.
can be seen in the file.

The arguments of 'inferir' are stored in "src/inferir_params.py". They are similar than
the mencionated before.
More details in "documentacion.pdf".
A typical configuration for all these parameters is provided with the software.


====================================
3- DATASET
====================================
Data files in "/data/" are generated using GenData tool.
Link to GenData project: 
https://github.com/sfenoglio/GenData


=================================
4- OUTPUT FILES
=================================
In 'salida/' are placed figures that contains graphics of loss, recall and jaccard per epoch,
for training and validation. In a '.npz' file are saved this metrics too.
Also, in 'salida/predicciones' are saved images of network outputs. These ones show the
prediction of the convolutional network on the left and the ground truth on the right.
The different colors correspond to the different classes.

In 'predicciones/' are saved the same metrics in a '.npz' file (if ground truth exists) and
images of network outputs.
When ground truth images don't exist, on the right can be seen the chromosome image.


=================================
5- EXAMPLES
=================================
'parameters.py' is configured to run a third stage of training of 'RedW.py'.
In 'salida/' there are the output files of the first and second stage.

'inferir_params.py' is configured to make inference with 'data/reales.npz' and 
the correspondent output is in 'inferencias/'. 

