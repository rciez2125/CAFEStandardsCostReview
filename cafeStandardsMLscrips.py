import numpy as np 
import pandas as pd 
import matplotlib
from sklearn.cluster import KMeans

def cleanSpreadsheet(sheetname):
	d = pd.read_excel('Inputs/FuelEconomyDatabase1975_2018.xlsx', sheet_name = sheetname)

	# create columns with year on year change in mpg 
	d['2-Cycle MPG Change'] = float(0)
	d['Real-World MPG Change'] = float(0)
	d['Real-World MPG_City Change'] = float(0)
	d['Real-World MPG_Hwy Change'] = float(0)
	d['RegulationDummy'] = 0
	d['RegulationExpected'] = 0

	# regulations passes: 1975, 1996 (LT), 2003 (LT), 2007 (EISA), 2012
	# regulation years: 1978, 1979 (trucks) - 1985, 1998-2003(LT), 2005-2007 2011-2016, 2017-2025
	regYears = [1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1998, 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
	regYearsAnnounced = [1975, 1976, 1977, 1996, 1997, 2003, 2004]
	regYearsAnnounced = regYearsAnnounced + regYears
	for n in range(d.shape[0]-1):
		d['2-Cycle MPG Change'][n+1] = d['2-Cycle MPG'][n+1] - d['2-Cycle MPG'][n]
		d['Real-World MPG Change'][n+1] = d['Real-World MPG'][n+1] - d['Real-World MPG'][n]
		d['Real-World MPG_City Change'][n+1] = d['Real-World MPG_City'][n+1] - d['Real-World MPG_City'][n]
		d['Real-World MPG_Hwy Change'][n+1] = d['Real-World MPG_Hwy'][n+1] - d['Real-World MPG_Hwy'][n]
		if (d['Model Year'][n+1]) in regYears:
			d['RegulationDummy'][n+1] = 1
		if d['Model Year'][n+1] in regYearsAnnounced:
			d['RegulationExpected'][n+1] = 1
	d['RegulationExpected'][0] = 1	



