import numpy as np
import pandas as pd
from cafeScripts import importInflation, import1975, import1980, import1990, import1999, import2007, import2017, interpolateCoeffs, linearInterpCoeffs, inflationConversion, applyLearningRate, interpWalk, makeFigure, importMPGdata, plotCostCurves, makeCompFigure, makePaperFig, makeSIFig
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
	inflatedCurves = inflationConversion(curves, baseYr)

	plotCostCurves(inflatedCurves, vehicleType, baseYr, 0, 1, 0, 1)
	plotCostCurves(inflatedCurves, vehicleType, baseYr, 1, 0, 0, 1)
	plotCostCurves(inflatedCurves, vehicleType, baseYr, 0, 0, 1, 1)
	plotCostCurves(inflatedCurves, vehicleType, baseYr, 1, 1, 1, 0)
	
	numSims = 1000
	costOutUni = np.zeros((76,5,numSims,3))
	for s in range(numSims):
		yearlyCoeffs_avg = interpWalk(inflatedCurves[:,:,1])
		costOutUni[:,:,s,1] = applyLearningRate(mpgImprove, yearlyCoeffs_avg)
		yearlyCoeffs_low = interpWalk(inflatedCurves[:,:,0])
		costOutUni[:,:,s,0] = applyLearningRate(mpgImprove, yearlyCoeffs_low)
		yearlyCoeffs_high = interpWalk(inflatedCurves[:,:,2])
		costOutUni[:,:,s,2] = applyLearningRate(mpgImprove, yearlyCoeffs_high)
	if vehicleType == "cars":
		figName = 'cumulativeCostCars.png'
	else:
		figName = 'cumulativeCostTrucks.png'
	makeFigure(costOutUni, figName, vehicleType, baseYr)
	print(costOutUni.shape)


	numSims = 1000
	costOutLin = np.zeros((76,5,numSims, 3))
	for s in range(numSims):
		yearlyCoeffs_avg = linearInterpCoeffs(inflatedCurves[:,:,1])
		costOutLin[:,:,s,1] = applyLearningRate(mpgImprove, yearlyCoeffs_avg)
		yearlyCoeffs_low = linearInterpCoeffs(inflatedCurves[:,:,0])
		costOutLin[:,:,s,0] = applyLearningRate(mpgImprove, yearlyCoeffs_low)
		yearlyCoeffs_high = linearInterpCoeffs(inflatedCurves[:,:,2])
		costOutLin[:,:,s,2] = applyLearningRate(mpgImprove, yearlyCoeffs_high)
	if vehicleType == "cars":
		figName = 'cumulativeCostCarsLinInterp.png'
	else:
		figName = 'cumulativeCostTrucksLinInterp.png'
	makeFigure(costOutLin, figName, vehicleType, baseYr)

	# simplify this 
	makeCompFigure(costOutUni, costOutLin, vehicleType, baseYr)

	if vehicleType == "cars":
		makePaperFig(inflatedCurves, costOutUni, vehicleType, baseYr)
	else:	
		makeSIFig(inflatedCurves, costOutUni, vehicleType, baseYr)

#runAnalysis("cars")
runAnalysis("trucks")

