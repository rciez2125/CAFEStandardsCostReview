import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def importInflation(): 
	data = pd.read_csv("inflationData.csv").values
	return(data)

def import1975():
	data = pd.read_csv("1975.csv").values	
	mod = sm.OLS(data[:,1],data[:,0])
	yhat = mod.fit()
	out = [1975, 1975, np.asscalar(yhat.params), 0, np.asscalar(yhat.bse), 0]
	return(out)

def import1980():
	data = pd.read_csv("1980.csv")
	endog = np.transpose(np.vstack((data.cumulativeMPG.values, data.cMPG2.values)))
	exog = data.cumulativeCost
	mod = sm.OLS(exog, endog)
	yhat = mod.fit()
	out = np.hstack((1980, 1980, yhat.params, yhat.bse))
	return(out)

def import1990(): 
	# return data for 1990 estimates
	data = pd.read_csv("1992Data.csv")	
	carSales = [100, 100, 100] #not right
	truckSales = [100, 100, 100] #not right
	# convert improvements from % to net improvement
	pi = data.EEA.values
	cost = data['4 Cyl'].values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)

	baselineMPG = 20 # not right
	data4Cyl = bcCalcOrder(pi/100, baselineMPG, cost)

	cost = data['6 Cyl'].values
	pi = data.EEA.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data6Cyl = bcCalcOrder(pi/100, baselineMPG, cost)

	cost = data['8 Cyl'].values
	pi = data.EEA.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data8Cyl = bcCalcOrder(pi/100, baselineMPG, cost)

	en1 = np.transpose(np.vstack((data4Cyl.dMPG, data4Cyl.dMPG2)))
	en2 = np.transpose(np.vstack((data6Cyl.dMPG, data6Cyl.dMPG2)))
	en3 = np.transpose(np.vstack((data8Cyl.dMPG, data8Cyl.dMPG2)))
	outData = np.zeros((2,6))
	outData[:,0] = 1990
	outData[:,1] = 1988
	outData[0, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl.cumulativeCost, data6Cyl.cumulativeCost, data8Cyl.cumulativeCost, carSales)
	outData[1, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl.cumulativeCost, data6Cyl.cumulativeCost, data8Cyl.cumulativeCost, truckSales)
	return outData

def import1999(): 
	carSales = [100, 100, 100] # not right
	truckSales = [2041, 3594, 1503] # not right


	print('hello world')

def import2007(): 
	carSales = [100, 100, 100] # not right
	truckSales = [2041, 3594, 1503] # not right
	data = pd.read_csv("2011Data.csv")

	pi = data.PercentI4Low.values
	cost = data.CostI4Low.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data4Cyl_low = bcCalcOrder(pi/100, baselineMPG, cost)

	pi = data.PercentV6Low.values
	cost = data.CostV6Low.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data6Cyl_low = bcCalcOrder(pi/100, baselineMPG, cost)

	pi = data.PercentV8Low.values
	cost = data.CostV8Low.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data8Cyl_low = bcCalcOrder(pi/100, baselineMPG, cost)

	outData = np.zeros((6,6))
	outData[:,0] = np.ones(6)*2009
	outData[:,1] = np.ones(6)*2008

	en1 = np.transpose(np.vstack((data4Cyl_low.dMPG, data4Cyl_low.dMPG2)))
	en2 = np.transpose(np.vstack((data6Cyl_low.dMPG, data6Cyl_low.dMPG2)))
	en3 = np.transpose(np.vstack((data8Cyl_low.dMPG, data8Cyl_low.dMPG2)))
	outData[0, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_low.cumulativeCost, data6Cyl_low.cumulativeCost, data8Cyl_low.cumulativeCost, carSales)
	outData[3, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_low.cumulativeCost, data6Cyl_low.cumulativeCost, data8Cyl_low.cumulativeCost, truckSales)

	# avg
	pi = data.PercentI4Avg.values
	cost = data.CostI4Avg.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data4Cyl_avg = bcCalcOrder(pi/100, baselineMPG, cost)

	pi = data.PercentV6Avg.values
	cost = data.CostV6Avg.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data6Cyl_avg = bcCalcOrder(pi/100, baselineMPG, cost)

	pi = data.PercentV8Avg.values
	cost = data.CostV8Avg.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data8Cyl_avg = bcCalcOrder(pi/100, baselineMPG, cost)

	en1 = np.transpose(np.vstack((data4Cyl_avg.dMPG, data4Cyl_avg.dMPG2)))
	en2 = np.transpose(np.vstack((data6Cyl_avg.dMPG, data6Cyl_avg.dMPG2)))
	en3 = np.transpose(np.vstack((data8Cyl_avg.dMPG, data8Cyl_avg.dMPG2)))
	outData[1, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_avg.cumulativeCost, data6Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, carSales)
	outData[4, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_avg.cumulativeCost, data6Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, truckSales)

	# high
	pi = data.PercentI4High.values
	cost = data.CostI4High.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data4Cyl_high = bcCalcOrder(pi/100, baselineMPG, cost)

	pi = data.PercentV6High.values
	cost = data.CostV6High.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data6Cyl_high = bcCalcOrder(pi/100, baselineMPG, cost)

	pi = data.PercentV8High.values
	cost = data.CostV8High.values
	dropList = np.array([])
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]):
			dropList = np.append(dropList, n)
	pi = np.delete(pi, dropList, axis = 0)
	cost = np.delete(cost, dropList, axis = 0)
	baselineMPG = 20 # not right
	data8Cyl_high = bcCalcOrder(pi/100, baselineMPG, cost)

	en1 = np.transpose(np.vstack((data4Cyl_high.dMPG, data4Cyl_high.dMPG2)))
	en2 = np.transpose(np.vstack((data6Cyl_high.dMPG, data6Cyl_high.dMPG2)))
	en3 = np.transpose(np.vstack((data8Cyl_high.dMPG, data8Cyl_high.dMPG2)))
	outData[2, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_high.cumulativeCost, data6Cyl_high.cumulativeCost, data8Cyl_high.cumulativeCost, carSales)
	outData[5, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_high.cumulativeCost, data6Cyl_high.cumulativeCost, data8Cyl_high.cumulativeCost, truckSales)
	return(outData)

def import2017(): 
	carSales = [7722, 1540, 253]
	truckSales = [2041, 3594, 1503] #4,6,8 cylinders
	# prices in 2010 data? 
	data4 = pd.read_csv("2017_4cyl.csv")
	data4['costAvg'] = (data4.costLow+data4.costHigh)/2
	data4['dMPGAvg'] = (data4.dMPGLow+data4.dMPGHigh)/2
	data4['dMPG2Avg'] = data4.dMPGAvg**2
	data6 = pd.read_csv("2017_6cyl.csv")
	data6['costAvg'] = (data6.costLow+data6.costHigh)/2
	data6['dMPGAvg'] = (data6.dMPGLow+data6.dMPGHigh)/2
	data6['dMPG2Avg'] = data6.dMPGAvg**2

	data8 = pd.read_csv("2017_8cyl.csv")
	data8['costAvg'] = (data8.costLow+data8.costHigh)/2
	data8['dMPGAvg'] = (data8.dMPGLow+data8.dMPGHigh)/2
	data8['dMPG2Avg'] = data8.dMPGAvg**2

	outData = np.zeros((6,6))
	outData[:,0] = np.ones(6)*2017
	outData[:,1] = np.ones(6)*2010

	carLowEn1 = np.transpose(np.vstack((data4.dMPGLow, data4.dMPG2Low)))
	carLowEn2 = np.transpose(np.vstack((data6.dMPGLow, data6.dMPG2Low)))
	carLowEn3 = np.transpose(np.vstack((data8.dMPGLow, data8.dMPG2Low)))

	carLowEn1 = np.transpose(np.vstack((data4.dMPGLow, data4.dMPG2Low)))
	carLowEn2 = np.transpose(np.vstack((data6.dMPGLow, data6.dMPG2Low)))
	carLowEn3 = np.transpose(np.vstack((data8.dMPGLow, data8.dMPG2Low)))
	outData[0, 2:6] = combinedCostFit(carLowEn1, carLowEn2, carLowEn3, data4.costLow, data6.costLow, data8.costLow, carSales)
	outData[3, 2:6] = combinedCostFit(carLowEn1, carLowEn2, carLowEn3, data4.costLow, data6.costLow, data8.costLow, truckSales)

	carLowEn1 = np.transpose(np.vstack((data4.dMPGAvg, data4.dMPG2Avg)))
	carLowEn2 = np.transpose(np.vstack((data6.dMPGAvg, data6.dMPG2Avg)))
	carLowEn3 = np.transpose(np.vstack((data8.dMPGAvg, data8.dMPG2Avg)))
	outData[1, 2:6] = combinedCostFit(carLowEn1, carLowEn2, carLowEn3, data4.costAvg, data6.costAvg, data8.costAvg, carSales)
	outData[4, 2:6] = combinedCostFit(carLowEn1, carLowEn2, carLowEn3, data4.costAvg, data6.costAvg, data8.costAvg, truckSales)

	carLowEn1 = np.transpose(np.vstack((data4.dMPGHigh, data4.dMPG2High)))
	carLowEn2 = np.transpose(np.vstack((data6.dMPGHigh, data6.dMPG2High)))
	carLowEn3 = np.transpose(np.vstack((data8.dMPGHigh, data8.dMPG2High)))
	outData[2, 2:6] = combinedCostFit(carLowEn1, carLowEn2, carLowEn3, data4.costHigh, data6.costHigh, data8.costHigh, carSales)
	outData[5, 2:6] = combinedCostFit(carLowEn1, carLowEn2, carLowEn3, data4.costHigh, data6.costHigh, data8.costHigh, truckSales)
	# return low, avg, high for cars and light trucks 
	return outData




	# return data for 1990 estimates

def import2025(): 
	# return data for 1990 estimates
	print('hello world')

def bcCalcOrder(percentImprovement, baselineMPG, costData):
	outData = pd.DataFrame()
	outData['percentMPG'] = percentImprovement
	outData['cost'] = costData
	outData['deltaMPG'] = percentImprovement * baselineMPG
	outData['bc'] = np.zeros(len(percentImprovement))
	
	for n in range(len(percentImprovement)):
		if costData[n] != 0:
			outData.bc[n] = outData.deltaMPG[n]/costData[n]
		else:
			outData.bc[n] = 1

	# sort data by benefit-cost ratio
	data = outData.sort_values(by = 'bc', ascending = False)
	data = data.reset_index(drop=True)
	data['cumulativeCost'] = np.zeros(len(percentImprovement))
	data['dMPG'] = np.zeros(len(percentImprovement))
	for n in range(len(percentImprovement)):
		data.cumulativeCost[n] = sum(data.cost[0:n+1])
		data.dMPG[n] = sum(data.deltaMPG[0:n+1])
	data['dMPG2'] = data.dMPG**2
	outData = data[['cumulativeCost', 'dMPG', 'dMPG2']]
	return(outData)

	# compute cumulative benefits and costs

	# return data 

	# calculate benefit-cost ratio for different technologies 
	# order the technologies by size of bc
	# calculate the cumulative mpg savings, cost
	# return data 


	print('helloworld')

def combinedCostFit(ex1, ex2, ex3, en1, en2, en3, carSales):
	mod1 = sm.OLS(en1, ex1)
	yhat1 = mod1.fit()
	mod2 = sm.OLS(en2, ex2)
	yhat2 = mod2.fit()
	mod3 = sm.OLS(en3, ex3)
	yhat3 = mod3.fit()

	yhat = np.zeros((1000,2))
	for n in range(1000):
		b1 = np.random.normal(yhat1.params[0], yhat1.bse[0])*carSales[0]/sum(carSales)
		b2 = np.random.normal(yhat2.params[0], yhat2.bse[0])*carSales[1]/sum(carSales)
		b3 = np.random.normal(yhat3.params[0], yhat3.bse[0])*carSales[2]/sum(carSales)
		yhat[n,0] = b1 + b2 + b3

		yhat[n,1] = (np.random.normal(yhat1.params[1], yhat1.bse[1])*carSales[0]/sum(carSales) + 
		np.random.normal(yhat2.params[1], yhat2.bse[1])*carSales[1]/sum(carSales) + np.random.normal(yhat3.params[1], yhat3.bse[1])*carSales[2]/sum(carSales))
	dataOut = np.hstack((np.average(yhat, axis = 0), stats.sem(yhat, axis=0)))
	return(dataOut)

def inflationConversion(curveCoeffs, baseYr):
	infData = importInflation()
	a = np.where(infData[:,0] == baseYr)
	for n in range(5):
		yr = curveCoeffs[n,1]
		m = np.where(infData[:,0] == yr)
		cf = infData[a,2]/infData[m,2]
		curveCoeffs[n, 2:6] = curveCoeffs[n,2:6]*cf
	return(curveCoeffs)

def interpolateCoeffs(curveCoeffs):
	yearlyCoeffs = np.ones((76, 3))
	yearlyCoeffs[:,0] = 1975 + np.linspace(0,75,76)
	yearlyCoeffs[0,1] = np.random.normal(curveCoeffs[0,2], curveCoeffs[0,4],1) 
	yearlyCoeffs[0,2] = np.random.normal(curveCoeffs[0,3], curveCoeffs[0,5],1)

	drawnCoeffs = np.ones((7,2))
	for n in range(7):
		drawnCoeffs[n,0] = np.random.normal(curveCoeffs[n,2], curveCoeffs[n,4],1)
		drawnCoeffs[n,1] = np.random.normal(curveCoeffs[n,3], curveCoeffs[n,5],1)
		
	for n in range(len(yearlyCoeffs)-1):
		if yearlyCoeffs[n+1,0] >= curveCoeffs[6,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[6,0:2]
		elif yearlyCoeffs[n+1,0]< curveCoeffs[1,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[0,0], curveCoeffs[1,0]-yearlyCoeffs[n+1,0], curveCoeffs[1,0]-curveCoeffs[0,0]]
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[0,0], drawnCoeffs[1,0]), max(drawnCoeffs[0,0], drawnCoeffs[1,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[0,1], drawnCoeffs[1,1]), max(drawnCoeffs[0,1], drawnCoeffs[1,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[1,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[1,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[2,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[1,0], curveCoeffs[2,0]-yearlyCoeffs[n+1,0], curveCoeffs[2,0]-curveCoeffs[1,0]]
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[2,0], drawnCoeffs[1,0]), max(drawnCoeffs[2,0], drawnCoeffs[1,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[2,1], drawnCoeffs[1,1]), max(drawnCoeffs[2,1], drawnCoeffs[1,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[2,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[2,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[3,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[2,0], curveCoeffs[3,0]-yearlyCoeffs[n+1,0], curveCoeffs[3,0]-curveCoeffs[2,0]]
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[2,0], drawnCoeffs[3,0]), max(drawnCoeffs[2,0], drawnCoeffs[3,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[2,1], drawnCoeffs[3,1]), max(drawnCoeffs[2,1], drawnCoeffs[3,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[3,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[3,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[4,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[3,0], curveCoeffs[4,0]-yearlyCoeffs[n+1,0], curveCoeffs[4,0]-curveCoeffs[3,0]]
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[4,0], drawnCoeffs[3,0]), max(drawnCoeffs[4,0], drawnCoeffs[3,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[4,1], drawnCoeffs[3,1]), max(drawnCoeffs[4,1], drawnCoeffs[3,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[4,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[4,:]
		elif yearlyCoeffs[n+1,0]==curveCoeffs[5,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[5,:]
		else:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[4,0], curveCoeffs[5,0]-yearlyCoeffs[n+1,0], curveCoeffs[5,0]-curveCoeffs[4,0]]
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[4,0], drawnCoeffs[5,0]), max(drawnCoeffs[4,0], drawnCoeffs[5,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[4,1], drawnCoeffs[5,1]), max(drawnCoeffs[4,1], drawnCoeffs[5,1]),1)
	return yearlyCoeffs

def applyLearningRate(mpgImprovementData, yearlyCoeffs):
	yearlyCosts = np.ones((76, 5))
	yearlyCosts[:,0] = 1975 + np.linspace(0,75,76)
	yearlyCosts[:,1] = mpgImprovementData

	yearlyCosts[:,2] = yearlyCosts[:,1]*yearlyCoeffs[:,1] + yearlyCoeffs[:,2]*yearlyCosts[:,1]**2
	LR = 0.02
	yearlyCosts[0,3:5] = 0
	for n in range(len(yearlyCosts)-1):
		yearlyCosts[n+1,3] = max(0, yearlyCosts[n+1,2]-yearlyCosts[n,2])
		yearlyCosts[n+1,4] = yearlyCosts[n,4]*(1-LR) + yearlyCosts[n+1,3]
	return(yearlyCosts)