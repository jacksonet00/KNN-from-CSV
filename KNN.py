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
fileIn = "data/" + input("Enter file name (with ext): ")

data = pd.read_csv(fileIn)

namesInCsv = pd.read_csv(fileIn, nrows=0)
namesClassified = []

for name in namesInCsv:
	namesClassified.append(name)

#Empty numpy array to replace empty columns
emptyColumn = []
emptyArray = np.asarray(emptyColumn)

#Future Update: This section will be of variable length
le = preprocessing.LabelEncoder()
#Five Data Points:
if len(data[namesClassified[0]]) != 0:
	itemAt0 = le.fit_transform((data[namesClassified[0]]))
else:
	itemAt0 = emptyArray
if len(data[namesClassified[1]]) != 0:
	itemAt1 = le.fit_transform((data[namesClassified[1]]))
else:
	itemAt1 = emptyArray
if len(data[namesClassified[2]]) != 0:
	itemAt2 = le.fit_transform((data[namesClassified[2]]))
else:
	itemAt2 = emptyArray
if len(data[namesClassified[3]]) != 0:
	itemAt3 = le.fit_transform((data[namesClassified[3]]))
else:
	itemAt3 = emptyArray
if len(data[namesClassified[4]]) != 0:
	itemAt4 = le.fit_transform((data[namesClassified[4]]))
else:
	itemAt4 = emptyArray
if len(data[namesClassified[5]]) != 0:
	itemAt5 = le.fit_transform((data[namesClassified[5]]))
else:
	itemAt5 = emptyArray
#Mandatory Prediction Column 7:
if len(data[namesClassified[6]]) != 0:
	itemAt6 = le.fit_transform((data[namesClassified[6]]))

predict = itemAt6

x = list(zip(itemAt0, itemAt1, itemAt2, itemAt3, itemAt4, itemAt5))
y = list(itemAt6)

#Allows user to partition the test group
testSize = input("Enter your test size: ")
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=float(testSize))

#Allows the user to specify the number of neighbors in the model
numNeighbors = input("Enter the number of neighbors to use: ")
model = KNeighborsClassifier(n_neighbors=int(numNeighbors))

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("The model had an accuracy of {}%" .format(round((acc * 100), 2)))

predicted = model.predict(x_test)
#Future Update: Names of prediction values will be stored in this list
names = []

#Allows user to visualize predictions
usrInExample = input("Would you like to see prediction data? (y/n) ")
if usrInExample == "y":
	usrInExampleSize = input(("Enter number of examples to display (less than {}): " .format(len(predicted[x_test]))))
	print("Example predictions:")
	for x in range((int(usrInExampleSize))):
		#Future Update: Prediction values will correspond with names in prediction column
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
