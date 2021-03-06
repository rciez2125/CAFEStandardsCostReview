import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd 

# make a plot of VMT vs fuel use and fuel economy vs gas price 

c_mpgFleet = [0.3,0,0.5]
c_mpgCar = [0.9, 0, 0]
c_mpgTruck = [0,0,0.9]
c_fuelConsumed = [0,0.5,0]
c_vmt = [0,0,0]
c_fuelSaved = [0.5,0.8,0.3]
c_fatalityRate = [1, 0.5, 0]
c_gasPrice = [0.9, 0.8, 0]

def makeFigVMTFuelEconGasPrices():
	# load some data
	data1 = pd.read_csv("Inputs/MPGSavingswRebound28Aug2019.csv")
	data2 = pd.read_csv("Inputs/NHTSACAFEData14Aug2019.csv")

	# make a mutliplot figure
	plt.figure(figsize=(7,4))
	ax1 = plt.subplot(1,2,1)
	#ax1.set_position([0.09, 0.15, 0.28, 0.8])

	ax1.plot(data1.Year, data1.VMTMilMiles/1000, '-', color = c_vmt)
	ax1.set_position([0.1, 0.15, 0.34, 0.8])
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.ylim(0,3300)
	plt.text(1998, data1.VMTMilMiles.values[-20]/1000+380, 'Vehicle Travel', FontSize = 7, color = c_vmt)
	plt.xlim(1965, 2018)
	plt.ylabel('Vehicle Miles [billions]', FontSize = 8)
	plt.xlabel('Years', FontSize = 8)


	ax2 = ax1.twinx()
	ax2.set_position([0.1, 0.15, 0.34, 0.8])
	ax2.plot(data1.Year, data1.FuelUseMil/1000, '-', color=c_fuelConsumed)
	ax2.tick_params(axis='both', which='major', labelsize=8, color = c_fuelConsumed, labelcolor=c_fuelConsumed)
	ax2.spines['right'].set_color(c_fuelConsumed)

	plt.ylim(0, 250)
	plt.text(1993, data1.FuelUseMil.values[-25]/1000-10, 'Fuel Consumption', FontSize = 7, color = c_fuelConsumed)
	plt.ylabel('Gallons [billions]', FontSize = 8, color = c_fuelConsumed)
	plt.xlim(1965, 2018)
	plt.text(2013, 240, 'a', FontSize = 8)
	plt.xlabel('Years', FontSize = 8)

	ax3 = plt.subplot(1,2,2)
	ax3.set_position([0.58, 0.15, 0.34, 0.8])
	plt.ylim(0, 45)
	ax3.plot(data2.Year, data2.CarMPG, '-', color = c_mpgCar)
	ax3.plot(data2.Year, data2.LTMPG, '-', color = c_mpgTruck)
	
	ax3.plot(data2.Year, data2.CarStd, '--', color = c_mpgCar)
	ax3.plot(data2.Year, data2.LTStd, '--', color = c_mpgTruck)
	
	ax3.plot(data2.Year, data2.EPACar, ':o', color = c_mpgCar)#change marker style
	ax3.plot(data2.Year, data2.EPALT, ':+', color=c_mpgTruck)# change marker style 
	#plt.text()
	ax3.tick_params(axis='both', which='major', labelsize=8)
	plt.xlim(1970, 2018)
	plt.ylabel('EPA Test Cycle Miles per Gallon', FontSize = 8)
	plt.xlabel('Year', FontSize = 8)
	plt.rc('legend',fontsize=8)
	plt.legend(('Car MPG', 'LT MPG', 'Car Standard', 'LT Standard', 'EPA Car', 'EPA LT'), ncol=2)

	ax4 = ax3.twinx()
	ax4.set_position([0.58, 0.15, 0.34, 0.8])
	ax4.plot(data2.Year, data2.GasPrices, '-', color = c_gasPrice) #change this color 
	plt.text(2000, 3.75, 'Gas Price', FontSize = 7, color = c_gasPrice)
	ax4.tick_params(axis='both', which='major', labelsize=8, color = c_gasPrice, labelcolor = c_gasPrice)
	ax4.spines['right'].set_color(c_gasPrice)
	plt.ylim(0, 4)
	plt.ylabel('Gas Prices [2018$/gallon]', FontSize = 8, color = c_gasPrice)
	plt.xlim(1970, 2018)
	plt.text(2013, 3.8, 'b', FontSize = 8)
	plt.xlabel('Year', FontSize = 8)

	plt.savefig('Figures/Figure2SideBySide.png', dpi = 300)
	plt.clf()

# on road fuel economy
def makeOnRoadMPGfig():
	data = pd.read_csv("Inputs/onRoadMPGData.csv")
	plt.figure(figsize=(3.33,3.33))
	ax1 = plt.subplot(1,1,1)
	ax1.set_position([0.13, 0.15, 0.69, 0.8])

	ax1.plot(data.ModelYear, data.EPA_Test, '-', color = c_mpgFleet)
	ax1.plot(data.ModelYear, data.EPA_Adjusted, '-', color = [0.1, 0.55, 0.6]) #change
	ax1.plot(data.ModelYear, data.FHWA_On_Road, '-', color = [0.62, 0.64, 0.11]) #change
	# add labels 
	plt.text(2004, 26, 'EPA Test', FontSize = 8, color = c_mpgFleet)
	plt.text(2007, 20.5, 'EPA\nAdjusted', color = [0.1, 0.55, 0.6], FontSize = 8, verticalalignment = 'top') #change
	plt.text(2000, 22.5, 'FHWA On Road', color = [0.62, 0.64, 0.11], FontSize = 8) #change
	plt.ylim(0,30)
	plt.xlim(1975, 2018)
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.xlabel('Model Year', FontSize = 8)
	plt.ylabel('Miles Per Gallon', FontSize = 8)

	ax2 = ax1.twinx()
	ax2.set_position([0.13, 0.15, 0.69, 0.8])
	ax2.set_yticks([1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10, 11.11, 12.5, 14.29, 16.67, 20, 25])# labels = (1, 0.5, 0.1, 0.05))
	ax2.tick_params(axis = 'both', which='major', labelsize=8)#
	ax2.set_yticklabels(['10$^0$', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '10$^{-1}$', ' ',' ',' ',' ',' ',' '])
	plt.ylim(0,30)
	plt.ylabel('Gallons Per Mile', FontSize = 8)

	plt.savefig('Figures/Figure3-2Axis.png', dpi = 300)
	plt.clf()

# miles of travel w/ rebound effect
def makeReboundFig():
	data1 = pd.read_csv("Inputs/MPGSavingswRebound28Aug2019.csv")

	# make a mutliplot figure
	plt.figure(figsize=(3.33,3.33))
	ax1 = plt.subplot(1,1,1)
	ax1.set_position([0.18, 0.15, 0.67, 0.8])

	ax1.plot(data1.Year, data1.VMTMilMiles/1000, '-', color = c_vmt)
	ax1.plot(data1.Year, data1.ReboundVMTMilMiles/1000, '--', color = [0.5, 0.5, 0.5])
	plt.ylim(0,3300)
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.xlim(1965, 2018)
	plt.text(1998, data1.VMTMilMiles.values[-20]/1000+380, 'Vehicle Travel', FontSize = 7, color = c_vmt)
	plt.text(1998, data1.ReboundVMTMilMiles.values[-20]/1000-320, 'Rebound\nRemoved', FontSize = 7, verticalalignment = 'bottom', color = [0.5, 0.5, 0.5])
	plt.ylabel('Vehicle Miles [billions]', FontSize = 8)
	plt.xlabel('Year', FontSize = 8)
	plt.xlim(1965, 2018)

	ax2 = ax1.twinx()
	ax2.set_position([0.18, 0.15, 0.67, 0.8])
	ax2.plot(data1.Year, data1.FuelUseMil/1000, '-', color = c_fuelConsumed)
	ax2.plot(data1.Year, data1.FuelSavingsMil/1000, '-.', color = c_fuelSaved)
	plt.text(1993, data1.FuelUseMil.values[-25]/1000-10, 'Fuel Consumption', FontSize = 7, color = c_fuelConsumed)
	plt.text(1993, data1.FuelSavingsMil.values[-25]/1000-10, 'Est. Fuel Savings', FontSize = 7, color = c_fuelSaved)
	ax2.tick_params(axis='both', which='major', labelsize=8, color = c_fuelConsumed, labelcolor = c_fuelConsumed)
	ax2.spines['right'].set_color(c_fuelConsumed)
	plt.ylim(0, 250)
	plt.ylabel('Gallons [billions]', FontSize = 8, color = c_fuelConsumed)
	plt.xlabel('Year', FontSize = 8)
	plt.xlim(1965, 2018)

	plt.savefig('Figures/Figure4.png', dpi = 300)
	plt.clf()

# cost figure: code in cafeScripts, runCostAnalysis

# motor vehicle and traffic fatalities (kick this to the SI?)
def makeFatalitiesFig():
	data = pd.read_csv("Inputs/FatalitiesFatalityRates.csv")

	pos1 = [0.1, 0.15, 0.44, 0.8]
	pos2 = [0.69, 0.15, 0.24, 0.8]
	plt.figure(figsize = (7, 3.33))
	plt.fontsize = 8
	ax1 = plt.subplot(1,2,1)
	ax1.set_position(pos1)
	ax1.plot(data.Year.values, data.TotalFatalities.values, '-', color = [0.2, 0.2, 0.2])
	ax1.plot([1985, 1985], [0, 60000], '--', color = [0.5, 0.5, 0.5])
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.text(1990, 30000, 'Total Fatalities', FontSize = 8, color = [0.2, 0.2, 0.2])
	plt.ylabel('Total Annual Fatalities', FontSize=8)
	plt.xlabel('Year', FontSize = 7)
	plt.xlim(1920, 2018)
	plt.ylim(0, 60000)

	ax2 = ax1.twinx()
	ax2.set_position(pos1)
	ax2.plot(data.Year, data.FatalityRate, '-', color = c_fatalityRate)
	plt.ylim(0,25)
	plt.text(1990, 3, 'Fatality Rate', FontSize = 8, color = c_fatalityRate)
	ax2.tick_params(axis='both', which='major', labelsize=8, color = c_fatalityRate, labelcolor = c_fatalityRate)
	ax2.spines['right'].set_color(c_fatalityRate)
	plt.ylabel('Fatalities per 100 Million Vehicle Miles', FontSize = 8, color = c_fatalityRate)
	plt.xlabel('Year', FontSize = 8)
	plt.xlim(1920, 2018)
	plt.text(2013, 23, 'a', FontSize = 8)

	#plt.savefig('FatalitiesFatalityRate.png', dpi = 300)
	#plt.clf()

	
	ax3 = plt.subplot(1,2,2)
	ax3.set_position(pos2)
	ax3.plot(data.Year, data.MPG, '-', color = c_mpgFleet)
	ax3.tick_params(axis='y', which='major', labelsize=8, color =c_mpgFleet, labelcolor = c_mpgFleet)
	ax3.spines['left'].set_color(c_mpgFleet)
	plt.ylabel('On-road Fleet Average Miles per Gallon', FontSize = 8, color = c_mpgFleet)
	plt.xlabel('Year', FontSize = 8)
	plt.xlim(1965, 2018)
	plt.text(1999, 18, 'MPG', FontSize = 8, color = c_mpgFleet)
	plt.ylim(0, 25)

	ax4 = ax3.twinx()
	ax4.set_position(pos2)
	ax4.plot(data.Year, data.FatalityRate, '-', color = c_fatalityRate)
	plt.text(1990, 3, 'Fatality Rate', FontSize = 8, color = c_fatalityRate)
	ax4.tick_params(axis='both', which='major', labelsize=8, color = c_fatalityRate, labelcolor = c_fatalityRate)
	ax4.spines['right'].set_color(c_fatalityRate)
	ax4.spines['left'].set_color(c_mpgFleet)
	plt.ylim(0,25)
	plt.ylabel('Fatalities per 100 Million Vehicle Miles', FontSize = 8, color = c_fatalityRate)
	plt.xlabel('Year', FontSize = 8)
	plt.text(2013, 23, 'b', FontSize = 8)
	plt.xlim(1965, 2018)

	plt.savefig('Figures/Fig7MPGFatalityRate.png', dpi = 300)
	plt.clf()

def makeMarginalBenefitCostFig():
	c = pd.read_excel('Inputs/MarginalCostvWTP31Mar2020.xlsx', sheet_name = 'Cars', skiprows = 2)
	print(c.columns)
	c = c.drop([0])
	c = c.reset_index(drop= True)
	c = c.rename(columns={"Unnamed: 0": "Year"})
	t = pd.read_excel('Inputs/MarginalCostvWTP31Mar2020.xlsx', sheet_name = 'Trucks', skiprows = 2)
	t = t.drop([0])
	t = t.reset_index(drop= True)
	t = t.rename(columns={"Unnamed: 0": "Year"})

	pos1 = [0.09, 0.13, 0.4, 0.8]
	pos2 = [0.57, 0.13, 0.4, 0.8]
	plt.figure(figsize = (7, 3.33))
	plt.fontsize = 8
	
	ax1 = plt.subplot(1,2,1)
	ax1.set_position(pos1)
	ax1.plot(c.Year, c['Marginal WTP'], '-', color = 'g') # fix colors
	ax1.plot(c.Year, c['MC Mean'], 'Dk', ms = 2)
	ax1.plot(c.Year, c['Moving Average'], '--k')
	ax1.fill_between(c.Year, c['MA High'], c['MA Low'], facecolor = 'k', alpha = 0.3)
	plt.xlim(1975, 2018)
	plt.ylim(-200, 1200)
	plt.ylabel('Dollars per 1  MPG Increase', fontsize = 8)
	plt.xticks(fontsize= 8)
	plt.yticks(fontsize= 8)
	plt.xlabel('Year', fontsize = 8)
	plt.text(2016, 1120, 'a', fontsize = 8)
	plt.legend(('Marginal Value', 'Mean MC', 'Mean MC Moving Average', 'MC Moving Avg Range'), fontsize = 6, loc= 'upper center')
	plt.title('Passenger Cars', fontsize = 8)

	ax2 = plt.subplot(1,2,2)
	ax2.set_position(pos2)
	ax2.plot(t.Year, t['Marginal WTP'], '-', color = 'g') # fix colors
	ax2.plot(t.Year, t['MC Mean'], 'Dk', ms = 2)
	ax2.plot(t.Year, t['Moving Average'], '--k')
	ax2.fill_between(t.Year, t['M.A. High'], t['M.A. Low'], facecolor = 'k', alpha = 0.3)
	plt.xlim(1975, 2018)
	plt.ylim(-200, 1200)
	plt.xticks(fontsize= 8)
	plt.yticks(fontsize= 8)
	plt.text(2016, 1120, 'b', fontsize = 8)
	plt.legend(('Marginal Value', 'Mean MC', 'Mean MC Moving Average', 'MC Moving Avg Range'), fontsize = 6)
	plt.xlabel('Year', fontsize = 8)
	plt.title('Light Trucks', fontsize = 8)

	plt.savefig('Figures/Fig6MarginalBenefits.png', dpi = 300)
	plt.clf()

# make the figures
# figure 1 is timeline figure 
#import timelineFigureNHTSAFlat

# figure 2 is 2 panels: trends in vehicle miles traveled and fuel consumption and differnet testing cycles
#makeFigVMTFuelEconGasPrices()

# figure 3 is on-road mpg 
#makeOnRoadMPGfig()

# figure 4 is a plot with vmt, fuel consumpiton, and rebound data 
#makeReboundFig()

# figure 5 is from the other analysis

# figure 6 is the marginal value and marginal cost 
#makeMarginalBenefitCostFig()

# figure 7 plots fatality data
#makeFatalitiesFig()

c = np.asarray([7.3,18.05,10.89156351,7.05640579, 23.89473082, 35.72010598, 35.74653434, 35.74653434])
d = c.clip(max = 20)
print(d)
