import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import pandas as pd 
import matplotlib.patches as mpatches
from cafeScripts import importMPGdata

# specify colors for different agencies
nhtsa = [0.55, 0.17, 0.89] # change from red to gold
epa = [0, 0.5, 0]
ca = [0.9, 0.8, 0.15] # change from blue to something else 



carcolor = [1, 0, 0] # aiming for a gold color --> this was read on plot 1
truckcolor = [0, 0, 1] #something purpleish  --> this was blue on plot 1 

data = pd.read_csv("NHTSACAFEData14Aug2019.csv")
car = data[data.Year > 1977]
truck = data[data.Year > 1978]

def addMPGdata(yr, mpg, nextMPG, ybottom, ls, regType, last):
	if regType == 'car':
		c = carcolor
	else: 
		c = truckcolor
	#r = plt.Rectangle((yr, ybottom), 1, mpg, color = c, alpha = a)
	plt.plot((yr, yr+1), (ybottom+mpg, ybottom+mpg), '-', alpha = ls, color = c)
	if last == 0:
		plt.plot((yr+1, yr+1), (ybottom+mpg, ybottom+nextMPG), '-', alpha = ls, color = c)
	#return r 

def makeAFigure():
	yrstr = 1970
	yrend = 2020
	xscale = 7/(2020-1970)
	#yfig = xcale*

	plt.figure(figsize=(7,3.5))
	plt.rcParams['text.usetex'] = False
	ax1 = plt.subplot(1,1,1)
	ax1.set_position([0, 0, 1, 1])

	y_nhtsa = -5
	y_epa = 0
	y_ca = 5

	
	x_merge = 2009

	fs = 6
	lineH = 0.5
	tickH = 1

	y_merged = y_nhtsa + lineH#y_epa 

	data = pd.read_csv("NHTSACAFEData14Aug2019.csv")
	car = data[data.Year > 1977]
	#car.CarStd = car.CarStd-car.CarStd.iloc[0]
	carNHTSA = car[car.Year<2009]
	carNHTSA.CarStd = carNHTSA.CarStd-car.CarStd.iloc[0]
	truck = data[data.Year > 1978]
	#truck.LTStd = truck.LTStd-truck.LTStd.iloc[0]
	truckNHTSA = truck[truck.Year<2009]
	truckNHTSA.LTStd = truckNHTSA.LTStd - truck.LTStd.iloc[0]
	
	carAll = car[car.Year>2008]
	print(carAll)
	carAll.CarStd = carAll.CarStd-car.CarStd.iloc[0]
	truckAll = truck[truck.Year>2008]
	truckAll.LTStd = truckAll.LTStd - truck.LTStd.iloc[0]
	
	x = car.CarStd.iloc[0] # pulled data from david's spreadsheet
	obama = [[2019, 41.4-x], [2020, 43.4-x], [2021, 45.2-x], [2022, 47.4-x], [2023, 49.7-x], [2024, 51.5-x], [2025, 54.8-x]]
	t = np.ones(6,)*obama[1][1]
	yr = np.linspace(2020, 2025, 6)
	carObama = pd.DataFrame(obama, columns = ['Year', 'CarStd']) 
	carTrump = pd.DataFrame(t, columns = ['CarStd'])
	carTrump['Year'] = yr

	x = truck.LTStd.iloc[0]
	obama = [[2019, 29.80-x], [2020, 30.80-x], [2021, 32.60-x], [2022, 33.90-x], [2023, 35.30-x], [2024, 37.50-x], [2025, 38.70-x]]
	truckObama = pd.DataFrame(obama, columns = ['Year', 'LTStd']) 
	t = np.ones(6,)*obama[1][1]
	yr = np.linspace(2020, 2025, 6)
	truckTrump = pd.DataFrame(t, columns = ['LTStd'])
	truckTrump['Year'] = yr

	prenhtsa_line = plt.Rectangle((1972, y_nhtsa-lineH/2),3, lineH, color = nhtsa, linewidth = 0, alpha = 0.4)
	nhtsa_line = plt.Rectangle((1975, y_nhtsa-lineH/2), (x_merge-1975), lineH, color = nhtsa, linewidth = 0, alpha = 1)
	nhtsa_connector = plt.Rectangle((x_merge-lineH/2, y_nhtsa+lineH/2), lineH, y_merged-y_nhtsa-2*lineH, color = nhtsa, linewidth=0)
	mergednhtsa_line = plt.Rectangle((x_merge, y_merged-3*lineH/2), (2026.5-x_merge), lineH, color = nhtsa, linewidth = 0, alpha = 1)
	plt.text(1971.5, y_nhtsa, 'NHTSA', color = nhtsa, fontsize = fs+2, horizontalalignment = 'right', verticalalignment = 'center')

	nhtsaW1 = mpatches.Wedge((x_merge-0.5, y_nhtsa+lineH/2), lineH, 270, 360, color = nhtsa, linewidth = 0)
	nhtsaW2 = mpatches.Wedge((x_merge+0.5, y_merged-3*lineH/2), lineH, 90, 180, color = nhtsa, linewidth = 0)

	def addNHTSADates():
		plt.plot((1973,1973),(y_nhtsa, y_nhtsa+tickH), '-k')
		plt.plot((1973.5, 1973.5), (y_nhtsa+tickH, y_nhtsa+11*tickH), '-k')
		plt.plot((1974,1974),(y_nhtsa, y_nhtsa+tickH), '-k')
		plt.plot((1973, 1974), (y_nhtsa+tickH, y_nhtsa+tickH), '-k')
		plt.text(1973.5, y_nhtsa+11.25*tickH, '1973-74: Oil Embargo', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'bottom')#, rotation = 90)

		plt.plot((1975, 1975), (y_nhtsa, y_nhtsa+8*tickH), '-k')
		plt.plot(1975, y_nhtsa, '.k')
		plt.text(1975-0.5, y_nhtsa+8.25*tickH, '1975: EPCA Passed', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'bottom')#, rotation = 90)

		plt.plot((1978, 1978), (y_nhtsa, y_nhtsa+5*tickH), '-k')
		plt.plot(1978, y_nhtsa, '.k')
		plt.text(1977, y_nhtsa+5.25*tickH, '1978: Passenger Car Standards\nbegin at ' + str(car.CarStd.iloc[0]) + ' mpg', fontsize = fs, verticalalignment = 'bottom')

		plt.plot((1979, 1979), (y_nhtsa, y_nhtsa-tickH), '-k')
		plt.plot(1979, y_nhtsa, '.k')
		plt.text(1978, y_nhtsa-1.25*tickH, '1979: Light Truck Standards\nbegin at ' + str(truck.LTStd.iloc[0]) + ' mpg', fontsize=fs, verticalalignment='top')

		plt.plot(1996, y_nhtsa, '.k') #light trucks only 
		plt.plot((1996, 1996), (y_nhtsa, y_nhtsa-2*lineH), '-k')
		plt.text(1998, y_nhtsa-2.25*lineH, '1996: New LT Standard,\n20.7 mpg', fontsize = fs, horizontalalignment = 'right', verticalalignment = 'top')

		plt.plot(2003, y_nhtsa, '.k')
		plt.plot((2003, 2003), (y_nhtsa, y_nhtsa-4*tickH), '-k')
		plt.text(2004, y_nhtsa-4.25*tickH, '2003: New LT Standard, 21.0 mpg\nby 2005, 22.2 mpg by 2007', fontsize = fs, horizontalalignment='right', verticalalignment = 'top')

		plt.plot((2007, 2007), (y_nhtsa, y_nhtsa-6*tickH), '-k')
		plt.plot(2007, y_nhtsa, '.k')
		plt.text(2007-0.5, y_nhtsa-6.25*tickH, '2007: EISA Passed, Set passenger car standard of 35.0\nmpg by 2020, required new NHTSA standard for\nlight trucks', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'top')#, rotation = 90)
	addNHTSADates()

	prepa_line = plt.Rectangle((2005, y_epa-lineH/2), 2009-2005-lineH, lineH, color = epa, linewidth = 0, alpha = 0.5)
	mergedepa_line = plt.Rectangle((x_merge+lineH/2, y_merged-lineH/2), (2026.5-x_merge-lineH/2), lineH, color = epa, linewidth = 0, alpha = 1)
	epaW1 = mpatches.Wedge((x_merge-lineH, y_epa-lineH/2), lineH, 0, 90, color = epa, linewidth = 0)
	epaW2 = mpatches.Wedge((x_merge+lineH, y_merged+3*lineH/2), lineH*2, 180, 270, color = epa, linewidth = 0)
	epa_connector = plt.Rectangle((x_merge-lineH, y_merged+lineH), lineH, y_epa-y_merged-lineH, color = epa, linewidth=0)

	def addEPADates():
		plt.plot(2009, y_epa, '.k')
		plt.plot(2007, y_epa, '.k')
		plt.plot((2007, 2007), (y_epa, y_epa+tickH), '-k')
		plt.plot((2009, 2007.5), (y_epa, y_epa+2.25*tickH), '-k')
		plt.text(2007, y_epa+1.25*tickH, '2007: $\it{Massachusetts v. EPA}$ Ruling', horizontalalignment = 'right', fontsize=fs, verticalalignment='bottom')
		plt.text(2007.5, y_epa+2.5*tickH, '2009: EPA Endangerment Finding', horizontalalignment = 'right', fontsize = fs, verticalalignment = 'bottom')
		plt.text(2004.5, y_epa, 'USEPA', horizontalalignment='right', verticalalignment = 'center', fontsize = fs + 2, color = epa)
	addEPADates()

	preca_line = plt.Rectangle((2007, y_ca-lineH/2), 2009-2007, lineH, color = ca, linewidth = 0, alpha = 0.5)
	#ca_line = plt.Rectangle((2005, y_ca-lineH/2), x_merge-2005-0.5, lineH, color = ca, linewidth = 0, alpha = 1)
	mergedca_line = plt.Rectangle((x_merge+0.5, y_merged+lineH/2), 2026.5-x_merge-0.5, lineH, color = ca, linewidth = 0, alpha = 1)
	caW1 = mpatches.Wedge((x_merge, y_ca-lineH/2), lineH, 0, 90, color = ca, linewidth = 0)
	caW2 = mpatches.Wedge((x_merge+lineH, y_merged + 3*lineH/2), lineH, 180, 270, color = ca, linewidth = 0)
	ca_connector = plt.Rectangle((x_merge, y_merged+3*lineH/2), lineH, y_ca-y_merged-2*lineH, color = ca, linewidth=0)
	plt.text(2006.5, y_ca, 'CARB', horizontalalignment='right', verticalalignment = 'center', color = ca, fontsize = fs+2)
	#plt.text(1999.5, y_ca+3*lineH/2, 'Anything in CA before they got their waiver in 2009 we should mention?', fontsize=fs, horizontalalignment='right')

	def addMergedDates():
		plt.plot(2012, y_merged, '.k')
		plt.plot((2012, 2012), (y_merged, y_merged-3*tickH), '-k')
		#plt.text(2011.5, y_merged-5.25*tickH, '2012 Obama Admin. Sets new standards\nfor 2017-2025, 54.5 mpg overall by 2025', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'top')
		plt.text(2011.5, y_merged-3.25*tickH, '2012: New standards for 2017-2025,\n54.5 mpg overall by 2025', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'top')	

		plt.plot(2018.67, y_merged, '.k')
		plt.plot((2018.67, 2018.67), (y_merged, y_merged+18*lineH), '-k')
		#plt.text(2018, y_merged + 7.25*lineH, 'August 2018 Trump\nAdmin. Proposed\nRollback, fix 2020\nstandards through 2026', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'bottom')
		plt.text(2018, y_merged + 18.25*lineH, 'August 2018: Trump\nAdmin. Proposed\nRollback', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'bottom')


		plt.plot((2009, 2009), (-7, 9), '-k')
		#plt.text(2008.5, 9.25+tickH, '2009 Obama Admin. National Program for\nharmonized standards between NHTSA, USEPA,\nCARB, 13 automakers', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'bottom')#, rotation = 90)
		plt.text(2008.5, 9.25, '2009: Obama Admin. National Program', fontsize = fs, horizontalalignment = 'left', verticalalignment = 'bottom')#, rotation = 90)
	addMergedDates()

	def addMPGBars():
		for n in range(len(carNHTSA.Year)):
			if n<len(carNHTSA.Year)-1:
				addMPGdata(carNHTSA.Year.iloc[n], carNHTSA.CarStd.iloc[n]/5, carNHTSA.CarStd.iloc[n+1]/5, y_nhtsa+lineH/2, 1, 'car', 0)
			else: 
				addMPGdata(carNHTSA.Year.iloc[n], carNHTSA.CarStd.iloc[n]/5, 0, y_nhtsa+lineH/2, 1, 'car',1)
			#ax1.add_patch(r)

		for n in range(len(carAll.Year)):
			if n<len(carAll.Year)-1:
				addMPGdata(carAll.Year.iloc[n], carAll.CarStd.iloc[n]/5, carAll.CarStd.iloc[n+1]/5, y_merged-lineH/2, 1, 'car', 0)
			else:
				addMPGdata(carAll.Year.iloc[n], carAll.CarStd.iloc[n]/5, 0, y_merged-lineH/2, 1, 'car', 1)
			#ax1.add_patch(r)

		for n in range(len(carObama.Year)):
			if n<len(carObama.Year)-1:
				if carObama.Year[n]<2020: 
					addMPGdata(carObama.Year.iloc[n], carObama.CarStd.iloc[n]/5, carObama.CarStd.iloc[n+1]/5, y_merged-lineH/2, 1, 'car', 0)
				else: 
					addMPGdata(carObama.Year.iloc[n], carObama.CarStd.iloc[n]/5, carObama.CarStd.iloc[n+1]/5, y_merged-lineH/2, 0.3, 'car', 0)
			else:
				addMPGdata(carObama.Year.iloc[n], carObama.CarStd.iloc[n]/5, 0, y_merged-lineH/2, 0.3, 'car', 1)
		plt.text(carObama.Year.iloc[-3]+0.25, carObama.CarStd.iloc[-3]/5+y_merged+2*lineH, 'Obama Rule', fontsize = fs-2, color = carcolor)

		for n in range(len(carTrump.Year)):
			#if n<len(carTrump.Year)-1:
			#addMPGdata(carTrump.Year.iloc[n], carTrump.CarStd.iloc[n]/5, carTrump.CarStd.iloc[n+1]/5, y_merged+3*lineH/2, 0.4, 'car', 1)
		#	else:
			addMPGdata(carTrump.Year.iloc[n], carTrump.CarStd.iloc[n]/5, 0, y_merged-lineH/2, 0.4, 'car', 1)
			#ax1.add_patch(r)
		plt.text(carTrump.Year.iloc[-3]+0.25, carTrump.CarStd.iloc[-3]/5+y_merged, 'Trump Proposal', fontsize = fs-2, color = carcolor)

		plt.plot((carObama.Year.iloc[0], carObama.Year.iloc[0]), (carObama.CarStd.iloc[0]/5+y_merged-lineH/2, carAll.CarStd.iloc[-1]/5+y_merged-lineH/2), color = carcolor, alpha = 1)

		for n in range(len(truckNHTSA.Year)):
			if n<len(truckNHTSA.Year)-1:
				addMPGdata(truckNHTSA.Year.iloc[n], truckNHTSA.LTStd.iloc[n]/5, truckNHTSA.LTStd.iloc[n+1]/5, y_nhtsa+lineH/2, 1, 'truck',0)
			else:
				addMPGdata(truckNHTSA.Year.iloc[n], truckNHTSA.LTStd.iloc[n]/5, 0, y_nhtsa+lineH/2, 1, 'truck',1)
			#ax1.add_patch(r)

		for n in range(len(truckAll.Year)):
			if n <len(truckAll.Year)-1:
				addMPGdata(truckAll.Year.iloc[n], truckAll.LTStd.iloc[n]/5, truckAll.LTStd.iloc[n+1]/5, y_merged-lineH/2, 1, 'truck', 0)
			else:
				addMPGdata(truckAll.Year.iloc[n], truckAll.LTStd.iloc[n]/5, 1, y_merged-lineH/2, 1, 'truck', 1)
			#ax1.add_patch(r)

		for n in range(len(truckObama.Year)):
			if n <len(truckObama.Year)-1:
				if truckObama.Year[n] < 2020:
					addMPGdata(truckObama.Year.iloc[n], truckObama.LTStd.iloc[n]/5, truckObama.LTStd.iloc[n+1]/5, y_merged-lineH/2, 1, 'truck',0)
				else: 
					addMPGdata(truckObama.Year.iloc[n], truckObama.LTStd.iloc[n]/5, truckObama.LTStd.iloc[n+1]/5, y_merged-lineH/2, 0.3, 'truck',0)
			else:
				addMPGdata(truckObama.Year.iloc[n], truckObama.LTStd.iloc[n]/5, 0, y_merged-lineH/2, 0.3, 'truck',1)
			#ax1.add_patch(r)
		plt.text(truckObama.Year.iloc[-3]+0.25, truckObama.LTStd.iloc[-3]/5+y_merged+lineH, 'Obama Rule', fontsize = fs-2, color = truckcolor)
		plt.plot((truckObama.Year.iloc[0], truckObama.Year.iloc[0]), (truckObama.LTStd.iloc[0]/5+y_merged+3*lineH/2, truckAll.LTStd.iloc[-1]/5+y_merged+3*lineH/2), color = truckcolor, alpha = 0.3)


		for n in range(len(truckTrump.Year)):
			addMPGdata(truckTrump.Year.iloc[n], truckTrump.LTStd.iloc[n]/5, 0, y_merged-lineH/2, 0.4, 'truck',1)
		plt.text(truckTrump.Year.iloc[-3]+0.25, truckTrump.LTStd.iloc[-3]/5+y_merged-2*lineH, 'Trump Proposal', fontsize = fs-2, color = truckcolor)
			#ax1.add_patch(r)
	addMPGBars()
	
	#ax1.plot(carAll.Year, y_merged + carAll.CarStd/5, '.k')
	#ax1.plot(truckNHTSA.Year, y_nhtsa-truckNHTSA.LTStd/5, '.b')
	#ax1.plot(truckAll.Year, y_merged-truckAll.LTStd/5, '.b')

	ax1.add_patch(prenhtsa_line)
	ax1.add_patch(nhtsa_line)
	ax1.add_patch(mergednhtsa_line)
	#ax1.add_patch(nhtsaW1)
	#ax1.add_patch(nhtsaW2)
	#ax1.add_patch(nhtsa_connector)

	ax1.add_patch(prepa_line)
	#ax1.add_patch(epa_line)
	ax1.add_patch(mergedepa_line)
	ax1.add_patch(epaW1)
	ax1.add_patch(epaW2)
	ax1.add_patch(epa_connector)

	ax1.add_patch(preca_line)
	#ax1.add_patch(ca_line)
	ax1.add_patch(mergedca_line)
	ax1.add_patch(caW1)
	ax1.add_patch(caW2)
	ax1.add_patch(ca_connector)

	plt.xlim(1971-3, 2031-3)
	plt.ylim(-15,15)

	#cc_pouch = Rectangle((-10, 0.1), 0.25, 0.4, color=cc_color,  linewidth=0)#, alpha=0.3)
	#ax.add_patch(cc_pouch)

	#matplotlib.patches.Patch(edgecolor=None, facecolor=None, color=None, linewidth=None, linestyle=None, antialiased=None, hatch=None, fill=True, capstyle=None, joinstyle=None, **kwargs)

	plt.savefig('TimelineFigureNHTSACenter.png', dpi = 300)
	plt.clf()

makeAFigure()