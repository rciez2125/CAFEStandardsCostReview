import numpy as np
import pandas as pd
from cafeScripts import importInflation, import1975, import1980, import1990, import1999, import2007, import2017, interpolateCoeffs, linearInterpCoeffs, inflationConversion, applyLearningRate, interpWalk, makeFigure, importMPGdata, plotCostCurves, makeCompFigure, makePaperFig, makeSIFig
from scipy import stats
# this will ultimately run the cost analysis for the CAFE standards review

# rough steps
# input raw technology cost data from NRC (and other?) studies
infData = importInflation()
#d07 = import2007("cars")
#d17 = import2017("cars", 2017)
#print(d17)

def runAnalysis(vehicleType):
	mpgImprove = importMPGdata(vehicleType).clip(0)
	#print(mpgImprove)
	curves = np.ones((8,7,3))
	baseYr = 2018

	data75 = import1975() # technology MY, currency yr, linear coeff, quad coeff, linear se, quad se] 
	data80 = import1980() 
	
	curves[0,:,:] = np.transpose(data75) #np.transpose(np.tile(data75,(3,1)))
	curves[1,:,:] = np.transpose(np.tile(data80,(3,1)))
	curves[2,:,:] = np.transpose(import1990(vehicleType))
	curves[3,:,:] = np.transpose(import1999(vehicleType))
	curves[4,:,:] = np.transpose(import2007(vehicleType))
	curves[5,:,:] = np.transpose(import2017(vehicleType, 2017))
	curves[6,:,:] = np.transpose(import2017(vehicleType, 2020))
	curves[7,:,:] = np.transpose(import2017(vehicleType, 2025))
	#print('2025 curves', curves[7,:,:])
	inflatedCurves = inflationConversion(curves, baseYr)
	#print(inflatedCurves)

	plotCostCurves(inflatedCurves, vehicleType, baseYr, 0, 1, 0, 1)
	plotCostCurves(inflatedCurves, vehicleType, baseYr, 1, 0, 0, 1)
	plotCostCurves(inflatedCurves, vehicleType, baseYr, 0, 0, 1, 1)
	plotCostCurves(inflatedCurves, vehicleType, baseYr, 1, 1, 1, 0)


	yearlyCoeffs = np.ones((76, 3))
	yearlyCoeffs[:,0] = 1975 + np.linspace(0,75,76)
	# data for 1975 (first year)
	yearlyCoeffs[0,1] = np.random.normal(inflatedCurves[0,2], inflatedCurves[0,4],1) 
	yearlyCoeffs[0,2] = np.random.normal(inflatedCurves[0,3], inflatedCurves[0,5],1)

	drawnCoeffs = np.ones((8,2))
	for n in range(8):
		drawnCoeffs[n,0] = np.random.normal(inflatedCurves[n,2], inflatedCurves[n,4],1)
		drawnCoeffs[n,1] = np.random.normal(inflatedCurves[n,3], inflatedCurves[n,5],1)
	#print('Curves')
	#print(inflatedCurves)
	#print('Drawn coeffs')
	#print(drawnCoeffs)

	
	numSims = 1000
	costOutUni = np.zeros((76,5,numSims,3))
	yrl_coeffs_data = np.zeros((76, 3, numSims))
	for s in range(numSims):
		yearlyCoeffs_avg = interpWalk(inflatedCurves[:,:,1])
		costOutUni[:,:,s,1] = applyLearningRate(mpgImprove, yearlyCoeffs_avg)
		yearlyCoeffs_low = interpWalk(inflatedCurves[:,:,0])
		costOutUni[:,:,s,0] = applyLearningRate(mpgImprove, yearlyCoeffs_low)
		yearlyCoeffs_high = interpWalk(inflatedCurves[:,:,2])
		costOutUni[:,:,s,2] = applyLearningRate(mpgImprove, yearlyCoeffs_high)
		yrl_coeffs_data[:,:,s] = yearlyCoeffs_avg
	if vehicleType == "cars":
		figName = 'Figures/cumulativeCostCars.png'
	else:
		figName = 'Figures/cumulativeCostTrucks.png'
	makeFigure(costOutUni, figName, vehicleType, baseYr)
	#print(yrl_coeffs_data.shape)

	# save some summary statistics
	#print(yrl_coeffs_data[:,1,:].shape)
	#print(np.mean(yrl_coeffs_data[:,1,:], axis = 0))
	#print(np.mean(yrl_coeffs_data[:,1,:], axis = 2))
	#print(np.mean(yrl_coeffs_data[:,1,:], axis = 2).shape)
	#print(np.mean(yrl_coeffs_data[:,2,:], axis = 1))
	
	#print(yrl_coeffs_data[:,:,0])
	#print(stats.sem(yrl_coeffs_data[:,1,:], axis = 1).shape)
	
	d = {'Linear Coeff Mean': np.mean(yrl_coeffs_data[:,1,:], axis = 1), 'Quad Coeff Mean': np.mean(yrl_coeffs_data[:,2,:], axis = 1), 
		'Linear Coeff Std Error': stats.sem(yrl_coeffs_data[:,1,:], axis = 1), 'Quad Coeff Std Error': stats.sem(yrl_coeffs_data[:,2,:], axis = 1),
		'Linear Coeff Std Dev': np.std(yrl_coeffs_data[:,1,:], axis = 1), 'Quad Coeff Std Dev': np.std(yrl_coeffs_data[:,2,:], axis = 1)}
	yr_means = pd.DataFrame(data = d, index = yrl_coeffs_data[:,0,0])
	yr_means.to_csv('MC_data'+vehicleType+'StdDev.csv')
	#yr_means = pd.DataFrame(data = np.mean(yrl_coeffs_data[:,1,:]), index = yrl_coeffs_data[:,0,0], columns = 'Linear Coefficient_Mean')
	#yr_means['Linear coefficient_mean'] = np.mean(yrl_coeffs_data[:,1,:])
	#print(yr_means)

	#	([yrl_coeffs_data[:,0,0], np.mean(yrl_coeffs_data[:,1,:], axis = 0), np.mean(yrl_coeffs_data[:,2,:], axis = 0)])
	#yr_means = np.mean(yrl_coeffs_data[:,1,:], axis = 0)
	#print(yr_means.shape) 


	#print(costOutUni.shape)
	#print(costOutUni[:,:,0,0])
	#d = pd.DataFrame(costOutUni)
	#d.to_csv('uniformOut.csv')

	numSims = 1000
	costOutLin = np.zeros((76,5,numSims,3))
	for s in range(numSims):
		yearlyCoeffs_avg = linearInterpCoeffs(inflatedCurves[:,:,1])
		costOutLin[:,:,s,1] = applyLearningRate(mpgImprove, yearlyCoeffs_avg)
		yearlyCoeffs_low = linearInterpCoeffs(inflatedCurves[:,:,0])
		costOutLin[:,:,s,0] = applyLearningRate(mpgImprove, yearlyCoeffs_low)
		yearlyCoeffs_high = linearInterpCoeffs(inflatedCurves[:,:,2])
		costOutLin[:,:,s,2] = applyLearningRate(mpgImprove, yearlyCoeffs_high)
	if vehicleType == "cars":
		figName = 'Figures/cumulativeCostCarsLinInterp.png'
	else:
		figName = 'Figures/cumulativeCostTrucksLinInterp.png'
	makeFigure(costOutLin, figName, vehicleType, baseYr)

	# simplify this 
	makeCompFigure(costOutUni, costOutLin, vehicleType, baseYr)

	if vehicleType == "cars":
		makePaperFig(inflatedCurves, costOutUni, vehicleType, baseYr)
	else:	
		makeSIFig(inflatedCurves, costOutUni, vehicleType, baseYr)

runAnalysis("cars")
#runAnalysis("trucks")

