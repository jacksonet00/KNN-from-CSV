import pandas as pd

names = pd.read_csv("car.data", nrows=0)
for name in names:
	print(name)
