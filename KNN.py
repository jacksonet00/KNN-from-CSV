import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

fileIn = input("Enter file name: ")

data = pd.read_csv(fileIn)

le = preprocessing.LabelEncoder()
buying = le.fit_transform((data["buying"]))
maint = le.fit_transform((data["maint"]))
door = le.fit_transform((data["door"]))
persons = le.fit_transform((data["persons"]))
lug_boot = le.fit_transform((data["lug_boot"]))
safety = le.fit_transform((data["safety"]))
cls = le.fit_transform((data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("The model had an accuracy of {}%" .format(round((acc * 100), 2)))

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

usrInExample = input("Would you like to see prediction data? (y/n) ")
if usrInExample == "y":
	usrInExampleSize = input(("Enter list size (less than {}): " .format(len(predicted[x_test]))))
	print("Example predictions:")
	for x in range((int(usrInExampleSize))):
	    print("Predicted: ", names[predicted[x]], "\t | \tActual: ", names[y_test[x]])

usrInSaveModel = input("Would you like to save this model? (y/n) ")
if usrInSaveModel == "y":
	model.save("model.sklearn")
