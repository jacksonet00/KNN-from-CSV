import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle
import csv

#User selects a .csv file from the data folder
fileIn = "data" + "\\" + input("Enter file name (with ext): ")

data = pd.read_csv(fileIn)

namesInCsv = pd.read_csv(fileIn, nrows=0)
namesClassified = []

for name in namesInCsv:
	namesClassified.append(name)

#set variables = items in csv
#currently hard coded
#TODO: Take row[0] of csv split on ","
#Then change data["buying"] to the item [0] in row[0]
le = preprocessing.LabelEncoder()
itemAt0 = le.fit_transform((data[namesClassified[0]]))
itemAt1 = le.fit_transform((data[namesClassified[1]]))
itemAt2 = le.fit_transform((data[namesClassified[2]]))
itemAt3 = le.fit_transform((data[namesClassified[3]]))
itemAt4 = le.fit_transform((data[namesClassified[4]]))
itemAt5 = le.fit_transform((data[namesClassified[5]]))
itemAt6 = le.fit_transform((data[namesClassified[6]]))

predict = "class"

#currently car.data items are hard coded
#TODO: modify this zip function to have items for whichever data points are pertanent
x = list(zip(itemAt0, itemAt1, itemAt2, itemAt3, itemAt4, itemAt5))
y = list(itemAt6)

#TODO: get user input to set the test_size for the training split
#give users a reccomended test size
testSize = input("Enter your test size: ")
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=float(testSize))

#TODO: allow users to specify the number of neighbors
#give users a reccomendation
numNeighbors = input("Enter the number of neighbors to use: ")
model = KNeighborsClassifier(n_neighbors=int(numNeighbors))

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("The model had an accuracy of {}%" .format(round((acc * 100), 2)))

predicted = model.predict(x_test)
#TODO: Find a way to store the actual prediction names in this list
names = ["unacc", "acc", "good", "vgood"]

usrInExample = input("Would you like to see prediction data? (y/n) ")
if usrInExample == "y":
	usrInExampleSize = input(("Enter list size (less than {}): " .format(len(predicted[x_test]))))
	print("Example predictions:")
	for x in range((int(usrInExampleSize))):
		#Predicted[x] & y_test[x] will show numerical representation of predictions
		#names[predicted[x]] will hold the name of the prediction (same with y_test)
	    print("Predicted: ", predicted[x], "\t | \tActual: ", y_test[x])

usrInSaveModel = input("Would you like to save this model? (y/n) ")
if usrInSaveModel == "y":
	filename = "stored_model.sav"
	pickle.dump(model, open(filename, 'wb'))


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