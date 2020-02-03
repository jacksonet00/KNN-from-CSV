# KNN from CSV
Build and save K Nearest Neighbors models from .csv files!
[![Run on Repl.it](https://repl.it/badge/github/jacksonet00/KNN-from-CSV)](https://repl.it/github/jacksonet00/KNN-from-CSV)

## Requirements.
* Sklearn
* Pandas
* Numpy

## Functionality.
KNN from CSV contains a main program (KNN.py) as well as two subfolders (data and models). When run, this program will allow the user to select a .csv file from the data folder and build a K Nearest Neighbors model of that data which will be stored in the models folder.

## Tutorial.
* Drag any number of .csv files into the data folder. This file should be 7 columns or less of data and the 7th column should contain the data to be predicted by the model.

* Run the file knn.py
* The program will ask for the name of the file you would like to model.
* The program will ask for the portion of the data to be used for testing the model. Ex: A value of 0.1 would partitian 10% of the data for testing the model's accuracy and use the remaining 90% of the data for training the model.
* The program will ask for the number of neighbors to use for the KNN model.
* The program will train a model of the data and report the accuracy of the model's predictions (this may be useful for determining if the user should retrain the model with a different number of neighbors or a different test partitian).
* The user may choose to visualize any number of tests from the accuracy testing.
* Finally, the user is given the option to save this model to the models folder.

## Loading a saved model.
This program allows the user to save a trained model using Python 3's pickle function. Once the model has been saved, it can be opened in Python using the following code:

```python
import pickle

pickle.open('models\example-file-name.sav')
```

## Future updates.
This product is currently version 1.0. The following updates and changes will come in future version of KNN from CSV:
* KNN from CSV will be able to handle .csv files with more than 7 columns.

* The user will be able to select a column to predict (currently this must be the 7th column, although columns 1-6 may be left blank).
* The prediction visualizer will be able to represent the names given in the .csv file rather than the model's numerical representation fo those values.
* The user will be able to retrain a model for improved accuracy without rerunning the program.
