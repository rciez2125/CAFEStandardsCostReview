import numpy as np 
import pandas as pd 
import matplotlib

d = pd.read_excel('FuelEconomyDatabase1975_2018.xlsx', sheet_name = 'ALL')
print(d.columns)

# create columns with year on year change in mpg 
d['2-Cycle MPG Change'] = float(0)
d['Real-World MPG Change'] = float(0)
d['Real-World MPG_City Change'] = float(0)
d['Real-World MPG_Hwy Change'] = float(0)

#for n in range(d.shape[0]-1):
#	d['2-Cycle MPG Change'][n+1] = d['2-Cycle MPG'][n+1] - d['2-Cycle MPG'][n]
#	d['Real-World MPG Change'][n+1] = d['Real-World MPG'][n+1] - d['Real-World MPG'][n]
#	d['Real-World MPG_City Change'][n+1] = d['Real-World MPG_City'][n+1] - d['Real-World MPG_City'][n]
#	d['Real-World MPG_Hwy Change'][n+1] = d['Real-World MPG_Hwy'][n+1] - d['Real-World MPG_Hwy'][n]

a = 267/(7.3*7.3)
print(a)
b = (267/7.3)-7.3*a
print(b)