import numpy as np
import matplotlib.pyplot as plt
from cafeScripts import importInflation, import1975, import1980, import1990, import2007, import2017, interpolateCoeffs, inflationConversion, applyLearningRate
# this will ultimately run the cost analysis for the CAFE standards review

# rough steps
# input raw technology cost data from NRC (and other?) studies
infData = importInflation()
data75 = import1975() # technology MY, currency yr, linear coeff, quad coeff, linear se, quad se] 
data80 = import1980() # done
data90 = import1990() # 2 rows, one for cars, one for trucks, sales data needs corrected
#data99 = import99()
data07 = import2007() # 6 rows, low, avg, high for cars and light trucks
data17 = import2017() # 6 rows, low, avg, high for cars and light trucks
#data25 # not necessary for now 

# create cost curves with uncertainty from the raw data
baseYr = 2015
curveCoeffs = np.ones((7, 6))
# 7 rows, 5 columns, corresponding with year, linear coeff, quadratic coeff, linear std error, quad std error
curveCoeffs[0,:] = data75 
curveCoeffs[1,:] = data80
curveCoeffs[2,:] = data90[0,:]#[1990, 118.891, 48.668, 0, 0] # the standard errors definitely aren't correct, I couldn't find these numbers anywhere
curveCoeffs[3,:] = [1999, 2010, 209.751, 3.788, 0, 0] # the standard errors definitely aren't correct, I'm confused by the calculation of the parameters in this model 
curveCoeffs[4,:] = data07[1,:] #cars avg
curveCoeffs[5,:] = data17[1,:] #cars avg 
curveCoeffs[6,:] = [2025, 2015, 24.349, 4.828, 4.443, 0.2146] # the standard errors definitely aren't correct

curveCoeffs_avg = inflationConversion(curveCoeffs, baseYr)
print(curveCoeffs_avg)

curveCoeffs = np.ones((7, 6))
# 7 rows, 5 columns, corresponding with year, linear coeff, quadratic coeff, linear std error, quad std error
curveCoeffs[0,:] = data75 
curveCoeffs[1,:] = data80
curveCoeffs[2,:] = data90[0,:]#[1990, 118.891, 48.668, 0, 0] # the standard errors definitely aren't correct, I couldn't find these numbers anywhere
curveCoeffs[3,:] = [1999, 2010, 209.751, 3.788, 0, 0] # the standard errors definitely aren't correct, I'm confused by the calculation of the parameters in this model 
curveCoeffs[4,:] = data07[0,:] #cars low
curveCoeffs[5,:] = data17[0,:] #cars low 
curveCoeffs[6,:] = [2025, 2015, 20.36, 3.53, 4.443, 0.2146] # the standard errors definitely aren't correct
curveCoeffs_low = inflationConversion(curveCoeffs, baseYr)

curveCoeffs = np.ones((7, 6))
# 7 rows, 5 columns, corresponding with year, linear coeff, quadratic coeff, linear std error, quad std error
curveCoeffs[0,:] = data75 
curveCoeffs[1,:] = data80
curveCoeffs[2,:] = data90[0,:]#[1990, 118.891, 48.668, 0, 0] # the standard errors definitely aren't correct, I couldn't find these numbers anywhere
curveCoeffs[3,:] = [1999, 2010, 209.751, 3.788, 0, 0] # the standard errors definitely aren't correct, I'm confused by the calculation of the parameters in this model 
curveCoeffs[4,:] = data07[2,:] #cars high
curveCoeffs[5,:] = data17[2,:] #cars high
curveCoeffs[6,:] = [2025, 2010, 24.349, 4.828, 4.443, 0.2146] # the standard errors definitely aren't correct
curveCoeffs_high = inflationConversion(curveCoeffs, baseYr)

numSims = 100
costOut_avg = np.zeros((76,5,numSims))
costOut_low = np.zeros((76,5,numSims))
costOut_high = np.zeros((76,5,numSims))

mpgImprovementData = [0, 1.66, 2.51, 4.09, 4.46, 7.69, 1.64, 2.55, 2.40, 2.79,
	3.45, 4.37, 4.54, 5.03, 4.59, 4.25, -0.06, -0.40, -0.15, -0.04, 
	0.33, 0.24, 0.40, 0.31, 0.01, -0.12, 0.15, 0.47, 0.86, 0.73, 
	1.28, 1.09, 2.02, 0.23, 1.56, 2.52, 2.28, 4.36, 5.45, 5.53, 
	8.25, 7.75, 8.75, 9.55, 11.35, 13.35, 15.15, 17.35, 19.65, 21.05,
	24.75, 25.05, 25.25, 25.25, 25.25, 25.35, 25.35, 25.25, 25.25, 25.35, 
	25.35, 25.35, 25.25, 25.25,	25.35, 25.45, 25.35, 25.35,	25.35, 25.35,
	25.25, 25.25, 25.25, 25.25, 25.15, 25.25] # delta MPG over the years
for s in range(numSims):
# replicate the weighted average curves
	yearlyCoeffs_avg = interpolateCoeffs(curveCoeffs_avg)
	costOut_avg[:,:,s] = applyLearningRate(mpgImprovementData, yearlyCoeffs_avg)

	yearlyCoeffs_low = interpolateCoeffs(curveCoeffs_low)
	costOut_low[:,:,s] = applyLearningRate(mpgImprovementData, yearlyCoeffs_low)

	yearlyCoeffs_high = interpolateCoeffs(curveCoeffs_high)
	costOut_high[:,:,s] = applyLearningRate(mpgImprovementData, yearlyCoeffs_high)

plotData = np.ones((76,10))
plotData[:,0] = 1975 + np.linspace(0,75,76)
for n in range(len(plotData)):
	plotData[n,1] = np.percentile(costOut_avg[n,4,:], 50)
	plotData[n,2] = np.percentile(costOut_avg[n,4,:], 2.5)
	plotData[n,3] = np.percentile(costOut_avg[n,4,:], 97.5)

	plotData[n,4] = np.percentile(costOut_low[n,4,:], 50)
	plotData[n,5] = np.percentile(costOut_low[n,4,:], 2.5)
	plotData[n,6] = np.percentile(costOut_low[n,4,:], 97.5)

	plotData[n,7] = np.percentile(costOut_high[n,4,:], 50)
	plotData[n,8] = np.percentile(costOut_high[n,4,:], 2.5)
	plotData[n,9] = np.percentile(costOut_high[n,4,:], 97.5)


plt.figure(figsize=(5,3))
plt.plot(plotData[:,0], plotData[:,1], '-b')
plt.fill_between(plotData[:,0], plotData[:,2], plotData[:,3], alpha = 0.5)

plt.plot(plotData[:,0], plotData[:,4], '-r')
plt.fill_between(plotData[:,0], plotData[:,5], plotData[:,6], alpha = 0.5)

plt.plot(plotData[:,0], plotData[:,7], '-g')
plt.fill_between(plotData[:,0], plotData[:,8], plotData[:,9], alpha = 0.5)

plt.xlim(1975,2018)



plt.ylim(0, 7000)
plt.text(1997, 5000, 'Pull from uniform distribution of different curve\ncoefficients. Missing some standard error information\nand only using high estimates', HorizontalAlignment = 'center')
plt.text(1997, 3500, 'Monte Carlo Simulations = '+str(numSims), HorizontalAlignment='center')
plt.ylabel('Cumulative Cost ['+str(baseYr)+'$]')
plt.xlabel('Year')
plt.savefig('CumulativeCosts.png', dpi=300)
