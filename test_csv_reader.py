import pandas as pd

data = pd.read_csv("data\\car.data")

names = pd.read_csv("data\\car.data", nrows=0)
namesClassified = []
for name in names:
	print(name)
	namesClassified.append(name)
print(namesClassified)

itemsAt0 = data[namesClassified[0]]
print(itemsAt0)

print(data["buying"])
