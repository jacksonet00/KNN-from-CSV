import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle
import csv

#Opening Message
print("Hello, you will find instructions for building your model in the README.md file.")

#User selects a .csv file from the data folder
fileIn = "data\\" + input("Enter file name (with ext): ")

data = pd.read_csv(fileIn)

namesInCsv = pd.read_csv(fileIn, nrows=0)
namesClassified = []

for name in namesInCsv:
	namesClassified.append(name)

#TODO: make this section variable length based on the number of rows of data
le = preprocessing.LabelEncoder()
if len(data[namesClassified[0]]) != 0:
	itemAt0 = le.fit_transform((data[namesClassified[0]]))
if len(data[namesClassified[1]]) != 0:
	itemAt1 = le.fit_transform((data[namesClassified[1]]))
if len(data[namesClassified[2]]) != 0:
	itemAt2 = le.fit_transform((data[namesClassified[2]]))
if len(data[namesClassified[3]]) != 0:
	itemAt3 = le.fit_transform((data[namesClassified[3]]))
if len(data[namesClassified[4]]) != 0:
	itemAt4 = le.fit_transform((data[namesClassified[4]]))
if len(data[namesClassified[5]]) != 0:
	itemAt5 = le.fit_transform((data[namesClassified[5]]))
if len(data[namesClassified[6]]) != 0:
	itemAt6 = le.fit_transform((data[namesClassified[6]]))

#TODO: Allow the user to specify the column that will be the prediction data (1-7) [x-1]
predict = data[namesClassified[6]]

#TODO: change this to data[namesClassified[x]] so that it will include all except predict
x = list(zip(itemAt0, itemAt1, itemAt2, itemAt3, itemAt4, itemAt5))
y = list(itemAt6)

#Sets test size of data based on user input
testSize = input("Enter your test size: ")
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=float(testSize))

#Sets number of neighbors based on user input
numNeighbors = input("Enter the number of neighbors to use: ")
model = KNeighborsClassifier(n_neighbors=int(numNeighbors))

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("The model had an accuracy of {}%" .format(round((acc * 100), 2)))

predicted = model.predict(x_test)
#TODO: Find a way to store the actual prediction names in this list
names = ["unacc", "acc", "good", "vgood"]

#Allows user to visualize predictions
usrInExample = input("Would you like to see prediction data? (y/n) ")
if usrInExample == "y":
	usrInExampleSize = input(("Enter number of examples to display (less than {}): " .format(len(predicted[x_test]))))
	print("Example predictions:")
	for x in range((int(usrInExampleSize))):
		#Predicted[x] & y_test[x] will show numerical representation of predictions
		#names[predicted[x]] will hold the name of the prediction (same with y_test)
	    print("Predicted: ", predicted[x], "\t | \tActual: ", y_test[x])

#Allows user to save the model to the models folder with a custom file name
usrInSaveModel = input("Would you like to save this model? (y/n) ")
if usrInSaveModel == "y":
	filename = "models\\" + input("Enter a file name to store model: ") + ".sav"
	with open(filename, 'wb') as f:
		pickle.dump(model, f)

#Closing Message
print("Thank you for using KNN from CSV.")
print("Remember to refer to the README.md file for more info on loading your saved models.")


"""
GENERAL NOTES ON FUNCTIONALITY
______________________________

Folder Structure:
.py file and README.md in base folder
models folder
data folder

program open with reminder to read instructions on README

users should upload csv files to data folder
generated models should be saved to models folder

program will ask which file you would like to use
file name is entered
program automatically appends path to choose the correct model

program will ask for test size (with recommendation)
program will ask for num_neighbors (with recommendation)

program will read out model complete with accuracy
program will ask if you would like to see prediction data
if yes it will ask number of examples

program will print prediction data examples

program will ask if you would like to save your model
if yes, program will ask user for a file name
model file will be saved to model folder

program end with message:
for more info on how to load and use your stored model
refer to the README document



"""