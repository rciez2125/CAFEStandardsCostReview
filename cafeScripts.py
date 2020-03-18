import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats

def importInflation(): 
	data = pd.read_csv("inflationData.csv").values
	return(data)

def importMPGdata(vehicleType):
	data = pd.read_csv("mpgImprovementData.csv")
	if vehicleType=="cars":
		return(data.PassengerCars.values)
	else:
		return(data.LightTrucks.values)

def import1975():
	a = 267/(7.3*7.3)
	b = (267/7.3)-7.3*a
	# this needs work 
	#data = pd.read_csv("1975.csv").values
	#mod = sm.OLS(data[:,2], data[:,0:1])
	#yhat = mod.fit()
	#out = np.hstack((1975, 1975, yhat.params, 0, yhat.bse, 0))
	upperBound = np.hstack((1975, 1975, 36.45, 0, 5.95, 0, 7.3))
	lowerBound = np.hstack((1975, 1975, b, a, 0, 2.44, 7.3))
	out = np.vstack((lowerBound, upperBound, upperBound))
	return(out)

def import1980():
	data = pd.read_csv("1980.csv")
	endog = np.transpose(np.vstack((data.cumulativeMPG.values, data.cMPG2.values)))
	exog = data.cumulativeCost
	mod = sm.OLS(exog, endog)
	yhat = mod.fit()
	out = np.hstack((1980, 1980, yhat.params, yhat.bse, data.cumulativeMPG.values[-1]))
	return(out)

def cleanData(pi, cost, techCat, abbreviations):
	df = pd.DataFrame(pi, columns = ['pi'])
	df['cost'] = cost
	df['techCat'] = techCat
	df['bc'] = df.pi/df.cost
	df['abbreviations'] = abbreviations
	df.abbreviations = df.abbreviations.str.strip()
	#print(abbreviations)
	#print('post strip', df.abbreviations)
	#print(df.head(10))

	dropList = np.array([])
	#transmissionComp = 0
	for n in range(len(df.cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]) or techCat[n] == "AltFuels":
			dropList = np.append(dropList, n)
	df = df.drop(dropList)
	df = df.reset_index(drop = True)

	t = df.index[df.techCat == 'Transmission'].tolist()
	if t: 
		tmax = df.index[df.bc[t[0]:t[-1]].idxmax()]
		t.remove(tmax)
		df = df.drop(t)
	df = df.reset_index(drop = True)

	# remove any duplicates 
	df = df.drop_duplicates(subset = ['pi', 'cost'], keep = 'first')
	df = df.reset_index(drop = True)

	# do the same thing for duplicate abbreviations 
	t = df.index[df.abbreviations == 'DVVL'].tolist()
	bcmax = df.bc.iloc[t]
	b = np.argmin(bcmax)
	df = df.drop(t)
	df = df.reset_index(drop = True)

	t = df.index[df.abbreviations == 'CVVL'].tolist()
	bcmax = df.bc.iloc[t]
	b = np.argmin(bcmax)
	df = df.drop(t)
	df = df.reset_index(drop = True)

	pi = np.asarray(df.pi)
	cost = np.asarray(df.cost)
	return(pi, cost)

def removeTechsByClass(df, techName):
	t = df.index[df.techCat == techName].tolist()
	g = df.index[df.groupby(['techCat'])['bc'].idxmax()]
	for n in range(len(g)):
		if g[n] in t:
			t.remove(g[n])
	df = df.drop(t)
	df = df.reset_index(drop = True)
	return(df)

def basicClean(pi, cost, techCat):
	df = pd.DataFrame(pi, columns = ['pi'])
	df['cost'] = cost
	df['techCat'] = techCat
	df['bc'] = df.pi/df.cost

	dropList = np.array([])
	for n in range(len(df.cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]) or techCat[n] == "AltFuels":
			dropList = np.append(dropList, n)
		if df.bc[n]>1 or df.bc[n]<0:
			df.bc[n] = 1
	df = df.drop(dropList)
	df = df.reset_index(drop = True)

	df = df.drop_duplicates(subset = ['pi', 'cost'], keep = 'first')
	df = df.reset_index(drop=True)
	return df

def clean90Data(pi, cost, techCat):
	df = basicClean(pi, cost, techCat)
	df = removeTechsByClass(df, 'Transmission')
	df = removeTechsByClass(df, 'Valve')
	df = removeTechsByClass(df, 'FuelSys')
	pi = np.asarray(df.pi)
	cost = np.asarray(df.cost)
	return(pi, cost)	

def clean99Data(pi, cost, techCat):
	df = pd.DataFrame(pi, columns = ['pi'])
	df['cost'] = cost
	df['techCat'] = techCat
	df['bc'] = df.pi/df.cost
	dropList = np.array([])

	for n in range(len(df.cost)):
		if pi[n] == 0:
			dropList = np.append(dropList, n)
	df = df.drop(dropList)
	df = df.reset_index(drop = True)

	#df = removeTechsByClass(df, 'Transmission')
	pi = np.asarray(df.pi)
	cost = np.asarray(df.cost)
	return(pi, cost)

def avg1990(outData):
	# monte carlo to get an average curve estimate because cost and % data are totally different 
	#for n in range(10): #make longer range later
	s = 1000
	low_linear = np.random.normal(outData[0,2], outData[0,4], s)
	low_quad = np.random.normal(outData[0,3], outData[0,5], s)
	high_linear = np.random.normal(outData[2,2], outData[2,4], s)
	high_quad = np.random.normal(outData[2,3], outData[2,5], s)
	avg_linear = (low_linear + high_linear)/2
	avg_quad = (low_quad + high_quad)/2 
	outData[1,2:6] = [np.mean(avg_linear), np.mean(avg_quad), stats.sem(avg_linear), stats.sem(avg_quad)]
	outData[1,6] = np.average((outData[0,6], outData[2,6]))
	return(outData)

def import1990(vehicleType): 
	# return data for 1990 estimates
	data = pd.read_csv("1992Data.csv")
	outData = np.zeros((3,7))
	outData[:,0] = 1990
	outData[:,1] = 1990

	if vehicleType == "cars":
		sales = [4357, 2655, 889]#[2048, 2185, 2011, 1033] # subcompact, compact, mid large
		baselineMPG = [31.25, 25.3, 22.6] #[31.4, 29.4, 26.1, 23.5] # table 8-3, subcompact, compact, mid, large
	else: 
		sales = [698, 2377, 929] #[1048, 576, 759, 1160] # small suv, small pickup, minivan, large pickup 
		baselineMPG = [26, 21.4, 17.1] #[21.3, 25.7, 22.8, 19.1] # small suv, small pickup, minivan, large pickup

	mpgLim = 200
	# convert improvements from % to net improvement
	# EEA cost estimates (low)
	pi, cost = clean90Data(data.EEA.values, data['4CylEEA'].values, data.TechCat)
	cost = cost *1.1 # convert to 1990 dollars because they are the worst
	#sc_low = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)
	c_low = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)

	pi, cost = clean90Data(data.EEA.values, data['6CylEEA'].values, data.TechCat)
	cost = cost*1.1
	ms_low = bcCalcOrder(pi/100, baselineMPG[1], cost, mpgLim)

	pi, cost = clean90Data(data.EEA.values, data['8CylEEA'].values, data.TechCat)
	cost = cost*1.1
	lg_low = bcCalcOrder(pi/100, baselineMPG[2], cost, mpgLim)

	#en1 = np.transpose(np.vstack((sc_low.dMPG, sc_low.dMPG2)))
	x2 = np.transpose(np.vstack((c_low.dMPG, c_low.dMPG2)))
	x3 = np.transpose(np.vstack((ms_low.dMPG, ms_low.dMPG2)))
	x4 = np.transpose(np.vstack((lg_low.dMPG, lg_low.dMPG2)))
	
	outData[0, 2:7] = combinedCostFit(x2, x3, x4, x4, x4, x4, c_low.cumulativeCost, ms_low.cumulativeCost, lg_low.cumulativeCost, lg_low.cumulativeCost, lg_low.cumulativeCost, lg_low.cumulativeCost, sales, 3)

	# SRI cost estimates (high)

	pi, cost = clean90Data(data.SRI.values, data['4CylSRI'].values, data.TechCat)
	#sc_high = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)
	c_high = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)

	pi, cost = clean90Data(data.SRI.values, data['6CylSRI'].values, data.TechCat)
	ms_high = bcCalcOrder(pi/100, baselineMPG[1], cost, mpgLim)

	pi, cost = clean90Data(data.SRI.values, data['8CylSRI'].values, data.TechCat)
	lg_high = bcCalcOrder(pi/100, baselineMPG[2], cost, mpgLim)

	#en1 = np.transpose(np.vstack((sc_high.dMPG, sc_high.dMPG2)))
	x2 = np.transpose(np.vstack((c_high.dMPG, c_high.dMPG2)))
	x3 = np.transpose(np.vstack((ms_high.dMPG, ms_high.dMPG2)))
	x4 = np.transpose(np.vstack((lg_high.dMPG, lg_high.dMPG2)))
	
	outData[2, 2:7] = combinedCostFit(x2, x3, x4, x4, x4, x4, c_high.cumulativeCost, ms_high.cumulativeCost, lg_high.cumulativeCost, lg_high.cumulativeCost, lg_high.cumulativeCost, lg_high.cumulativeCost, sales, 3)

	outData = avg1990(outData)
	return outData

def import1999(vehicleType): 

	data = pd.read_csv("1999.csv")
	
	outData = np.zeros((3,7))
	outData[:,0] = np.ones(3)*1999# confirmed
	outData[:,1] = np.ones(3)*2000# dont' know if this is right

	if vehicleType == "cars":
		sales = [1381, 2377, 3141, 1059] #EPA data through 2003, appendix g 
		#[1321, 1998, 2969, 1179] # from spreadsheet, subcompact, compact, midsize, large
		baselineMPG = [31.3, 30.1, 27.1, 24.8] 
		mpgLim = 200
		# low performance, high cost 
		pi_sc, cost_sc = clean99Data(data.FCE_low.values*data.Subcompact_high.values, data.RPE_high.values, data.TechCat)
		sc_low = bcCalcOrder(pi_sc/100, baselineMPG[0], cost_sc, mpgLim)
		pi_c, cost_c = clean99Data(data.FCE_low.values*data.Compact_high.values, data.RPE_high.values, data.TechCat)
		c_low = bcCalcOrder(pi_c/100, baselineMPG[1], cost_c, mpgLim)
		pi_ms, cost_ms = clean99Data(data.FCE_low.values*data.Midsize_high.values, data.RPE_high.values, data.TechCat)
		ms_low = bcCalcOrder(pi_ms/100, baselineMPG[2], cost_ms, mpgLim)
		pi_lg, cost_lg = clean99Data(data.FCE_low.values*data.Large_high.values, data.RPE_high.values, data.TechCat)
		lg_low = bcCalcOrder(pi_lg/100, baselineMPG[3], cost_lg, mpgLim)	

		carLowx1 = np.transpose(np.vstack((sc_low.dMPG, sc_low.dMPG2)))
		carLowx2 = np.transpose(np.vstack((c_low.dMPG, c_low.dMPG2)))
		carLowx3 = np.transpose(np.vstack((ms_low.dMPG, ms_low.dMPG2)))
		carLowx4 = np.transpose(np.vstack((lg_low.dMPG, lg_low.dMPG2)))

		# avg cost and performance, avg pathway 
		pi_sc, cost_sc = clean99Data(data.FCE_avg.values*data.Subcompact_avg.values, data.RPE_avg.values, data.TechCat)
		sc_avg = bcCalcOrder(pi_sc/100, baselineMPG[0], cost_sc, mpgLim)
		pi_c, cost_c = clean99Data(data.FCE_avg.values*data.Compact_avg.values, data.RPE_avg.values, data.TechCat)
		c_avg = bcCalcOrder(pi_c/100, baselineMPG[1], cost_c, mpgLim)
		pi_ms, cost_ms = clean99Data(data.FCE_avg.values*data.Midsize_avg.values, data.RPE_avg.values, data.TechCat)
		ms_avg = bcCalcOrder(pi_ms/100, baselineMPG[2], cost_ms, mpgLim)
		pi_lg, cost_lg = clean99Data(data.FCE_avg.values*data.Large_avg.values, data.RPE_avg.values, data.TechCat)
		lg_avg = bcCalcOrder(pi_lg/100, baselineMPG[3], cost_lg, mpgLim)	

		carAvgx1 = np.transpose(np.vstack((sc_avg.dMPG, sc_avg.dMPG2)))
		carAvgx2 = np.transpose(np.vstack((c_avg.dMPG, c_avg.dMPG2)))
		carAvgx3 = np.transpose(np.vstack((ms_avg.dMPG, ms_avg.dMPG2)))
		carAvgx4 = np.transpose(np.vstack((lg_avg.dMPG, lg_avg.dMPG2)))

		#
		pi_sc, cost_sc = clean99Data(data.FCE_high.values*data.Subcompact_low.values, data.RPE_low.values, data.TechCat)
		sc_high = bcCalcOrder(pi_sc/100, baselineMPG[0], cost_sc, mpgLim)
		pi_c, cost_c = clean99Data(data.FCE_high.values*data.Compact_low.values, data.RPE_low.values, data.TechCat)
		c_high = bcCalcOrder(pi_c/100, baselineMPG[1], cost_c, mpgLim)
		pi_ms, cost_ms = clean99Data(data.FCE_high.values*data.Midsize_low.values, data.RPE_low.values, data.TechCat)
		ms_high = bcCalcOrder(pi_ms/100, baselineMPG[2], cost_ms, mpgLim)
		pi_lg, cost_lg = clean99Data(data.FCE_high.values*data.Large_low.values, data.RPE_low.values, data.TechCat)
		lg_high = bcCalcOrder(pi_lg/100, baselineMPG[3], cost_lg, mpgLim)	

		carHighx1 = np.transpose(np.vstack((sc_high.dMPG, sc_high.dMPG2)))
		carHighx2 = np.transpose(np.vstack((c_high.dMPG, c_high.dMPG2)))
		carHighx3 = np.transpose(np.vstack((ms_high.dMPG, ms_high.dMPG2)))
		carHighx4 = np.transpose(np.vstack((lg_high.dMPG, lg_high.dMPG2)))

		outData[2, 2:7] = combinedCostFit(carLowx1, carLowx2, carLowx3, carLowx4, carLowx4, carLowx4, sc_low.cumulativeCost, c_low.cumulativeCost, ms_low.cumulativeCost, lg_low.cumulativeCost, ms_low.cumulativeCost, ms_low.cumulativeCost, sales, 4)
		outData[1, 2:7] = combinedCostFit(carAvgx1, carAvgx2, carAvgx3, carAvgx4, carAvgx4, carAvgx4, sc_avg.cumulativeCost, c_avg.cumulativeCost, ms_avg.cumulativeCost, lg_avg.cumulativeCost, ms_avg.cumulativeCost, ms_avg.cumulativeCost, sales, 4)
		outData[0, 2:7] = combinedCostFit(carHighx1, carHighx2, carHighx3, carHighx4, carHighx4, carHighx4, sc_high.cumulativeCost, c_high.cumulativeCost, ms_high.cumulativeCost, lg_high.cumulativeCost, ms_high.cumulativeCost, ms_high.cumulativeCost, sales, 4)
	else: 
		sales = [314, 1762, 754, 1463, 213, 1571] #EPA through 2003, appendix H, don't count midsize trucks 
		#[300, 1716, 915, 1341, 232, 1418] # from spreadsheets, small suv, mid suv, large suv, minivan, small pickup, large pickup 
		baselineMPG = [24.1, 21, 17.2, 23, 23.2, 18.5]
		mpgLim = 200

		# high performance, low cost
		pi_ss, cost_ss = clean99Data(data.FCE_high.values*data.SmallSUV_low.values, data.RPE_low.values, data.TechCat)
		ss_low = bcCalcOrder(pi_ss/100, baselineMPG[0], cost_ss, mpgLim)
		pi_ms, cost_ms = clean99Data(data.FCE_high.values*data.MidSUV_low.values, data.RPE_low.values, data.TechCat)
		ms_low = bcCalcOrder(pi_ms/100, baselineMPG[2], cost_ms, mpgLim)
		pi_ls, cost_ls = clean99Data(data.FCE_high.values*data.LargeSUV_low.values, data.RPE_low.values, data.TechCat)
		ls_low = bcCalcOrder(pi_ls/100, baselineMPG[2], cost_ls, mpgLim)
		pi_mv, cost_mv = clean99Data(data.FCE_high.values*data.Minivan_low.values, data.RPE_low.values, data.TechCat)
		mv_low = bcCalcOrder(pi_mv/100, baselineMPG[2], cost_mv, mpgLim)
		pi_sp, cost_sp = clean99Data(data.FCE_high.values*data.SmallPickup_low.values, data.RPE_low.values, data.TechCat)
		sp_low = bcCalcOrder(pi_sp/100, baselineMPG[2], cost_sp, mpgLim)
		pi_lp, cost_lp = clean99Data(data.FCE_high.values*data.LargePickup_low.values, data.RPE_low.values, data.TechCat)
		lp_low = bcCalcOrder(pi_lp/100, baselineMPG[2], cost_lp, mpgLim)

		# avg cost and performance 
		pi_ss, cost_ss = clean99Data(data.FCE_avg.values*data.SmallSUV_avg.values, data.RPE_avg.values, data.TechCat)
		ss_avg = bcCalcOrder(pi_ss/100, baselineMPG[0], cost_ss, mpgLim)
		pi_ms, cost_ms = clean99Data(data.FCE_avg.values*data.MidSUV_avg.values, data.RPE_avg.values, data.TechCat)
		ms_avg = bcCalcOrder(pi_ms/100, baselineMPG[2], cost_ms, mpgLim)
		pi_ls, cost_ls = clean99Data(data.FCE_avg.values*data.LargeSUV_avg.values, data.RPE_avg.values, data.TechCat)
		ls_avg = bcCalcOrder(pi_ls/100, baselineMPG[2], cost_ls, mpgLim)
		pi_mv, cost_mv = clean99Data(data.FCE_avg.values*data.Minivan_avg.values, data.RPE_avg.values, data.TechCat)
		mv_avg = bcCalcOrder(pi_mv/100, baselineMPG[2], cost_mv, mpgLim)
		pi_sp, cost_sp = clean99Data(data.FCE_avg.values*data.SmallPickup_avg.values, data.RPE_avg.values, data.TechCat)
		sp_avg = bcCalcOrder(pi_sp/100, baselineMPG[2], cost_sp, mpgLim)
		pi_lp, cost_lp = clean99Data(data.FCE_avg.values*data.LargePickup_avg.values, data.RPE_avg.values, data.TechCat)
		lp_avg = bcCalcOrder(pi_lp/100, baselineMPG[2], cost_lp, mpgLim)

		# low performance, high cost
		pi_ss, cost_ss = clean99Data(data.FCE_low.values*data.SmallSUV_high.values, data.RPE_high.values, data.TechCat)
		ss_high = bcCalcOrder(pi_ss/100, baselineMPG[0], cost_ss, mpgLim)
		pi_ms, cost_ms = clean99Data(data.FCE_low.values*data.MidSUV_high.values, data.RPE_high.values, data.TechCat)
		ms_high = bcCalcOrder(pi_ms/100, baselineMPG[2], cost_ms, mpgLim)
		pi_ls, cost_ls = clean99Data(data.FCE_low.values*data.LargeSUV_high.values, data.RPE_high.values, data.TechCat)
		ls_high = bcCalcOrder(pi_ls/100, baselineMPG[2], cost_ls, mpgLim)
		pi_mv, cost_mv = clean99Data(data.FCE_low.values*data.Minivan_high.values, data.RPE_high.values, data.TechCat)
		mv_high = bcCalcOrder(pi_mv/100, baselineMPG[2], cost_mv, mpgLim)
		pi_sp, cost_sp = clean99Data(data.FCE_low.values*data.SmallPickup_high.values, data.RPE_high.values, data.TechCat)
		sp_high = bcCalcOrder(pi_sp/100, baselineMPG[2], cost_sp, mpgLim)
		pi_lp, cost_lp = clean99Data(data.FCE_low.values*data.LargePickup_high.values, data.RPE_high.values, data.TechCat)
		lp_high = bcCalcOrder(pi_lp/100, baselineMPG[2], cost_lp, mpgLim)

		carLowx1 = np.transpose(np.vstack((ss_low.dMPG, ss_low.dMPG2)))
		carLowx2 = np.transpose(np.vstack((ms_low.dMPG, ms_low.dMPG2)))
		carLowx3 = np.transpose(np.vstack((ls_low.dMPG, ls_low.dMPG2)))
		carLowx4 = np.transpose(np.vstack((mv_low.dMPG, mv_low.dMPG2)))
		carLowx5 = np.transpose(np.vstack((sp_low.dMPG, sp_low.dMPG2)))
		carLowx6 = np.transpose(np.vstack((lp_low.dMPG, lp_low.dMPG2)))

		carAvgx1 = np.transpose(np.vstack((ss_avg.dMPG, ss_avg.dMPG2)))
		carAvgx2 = np.transpose(np.vstack((ms_avg.dMPG, ms_avg.dMPG2)))
		carAvgx3 = np.transpose(np.vstack((ls_avg.dMPG, ls_avg.dMPG2)))
		carAvgx4 = np.transpose(np.vstack((mv_avg.dMPG, mv_avg.dMPG2))) 
		carAvgx5 = np.transpose(np.vstack((sp_avg.dMPG, sp_avg.dMPG2)))
		carAvgx6 = np.transpose(np.vstack((lp_avg.dMPG, lp_avg.dMPG2)))

		carHighx1 = np.transpose(np.vstack((ss_high.dMPG, ss_high.dMPG2)))
		carHighx2 = np.transpose(np.vstack((ms_high.dMPG, ms_high.dMPG2)))
		carHighx3 = np.transpose(np.vstack((ls_high.dMPG, ls_high.dMPG2)))
		carHighx4 = np.transpose(np.vstack((mv_high.dMPG, mv_high.dMPG2))) 
		carHighx5 = np.transpose(np.vstack((sp_high.dMPG, sp_high.dMPG2)))
		carHighx6 = np.transpose(np.vstack((lp_high.dMPG, lp_high.dMPG2)))

		outData[0, 2:7] = combinedCostFit(carLowx1, carLowx2, carLowx3, carLowx4, carLowx5, carLowx6, 
			ss_low.cumulativeCost, ms_low.cumulativeCost, ls_low.cumulativeCost, mv_low.cumulativeCost, sp_low.cumulativeCost, lp_low.cumulativeCost, sales, 6)
		outData[1, 2:7] = combinedCostFit(carAvgx1, carAvgx2, carAvgx3, carAvgx4, carAvgx5, carAvgx6,
			ss_avg.cumulativeCost, ms_avg.cumulativeCost, ls_avg.cumulativeCost, mv_avg.cumulativeCost, sp_avg.cumulativeCost, lp_avg.cumulativeCost, sales, 6)
		outData[2, 2:7] = combinedCostFit(carHighx1, carHighx2, carHighx3, carHighx4, carHighx5, carHighx6, 
			ss_high.cumulativeCost, ms_high.cumulativeCost, ls_high.cumulativeCost, mv_high.cumulativeCost, sp_high.cumulativeCost, lp_high.cumulativeCost, sales, 6)
	#data = pd.read_csv("1991Data.csv") this doesn't exist yet 
	return outData

def import2007(vehicleType): 
	if vehicleType == "cars":
		sales = [4975, 3493, 516] # from EPA2016 data
		baselineMPG = [34.04, 26.56, 22.7] # from EPA 2016 Data, generally lower than David's estimates
	else: 
		sales = [607, 3238, 2430] # from epa spreadsheet
		baselineMPG = [27.52, 22.69, 19.38] # epa combined
	data = pd.read_csv("2011Data.csv")

	mpgLim = 100

	#low performance, high cost 
	pi, cost = cleanData(data.PercentI4Low.values, data.CostI4High.values*1.5, data.TechCat, data.Abbreviation)
	data4Cyl_low = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)

	pi, cost = cleanData(data.PercentV6Low.values, data.CostV6High.values*1.5, data.TechCat, data.Abbreviation)
	data6Cyl_low = bcCalcOrder(pi/100, baselineMPG[1], cost, mpgLim)

	pi, cost = cleanData(data.PercentV8Low.values, data.CostV8High.values*1.5, data.TechCat, data.Abbreviation)
	data8Cyl_low = bcCalcOrder(pi/100, baselineMPG[2], cost, mpgLim)

	outData = np.zeros((3,7))
	outData[:,0] = np.ones(3)*2009
	outData[:,1] = np.ones(3)*2008

	x1 = np.transpose(np.vstack((data4Cyl_low.dMPG, data4Cyl_low.dMPG2)))
	x2 = np.transpose(np.vstack((data6Cyl_low.dMPG, data6Cyl_low.dMPG2)))
	x3 = np.transpose(np.vstack((data8Cyl_low.dMPG, data8Cyl_low.dMPG2)))
	outData[2, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, data4Cyl_low.cumulativeCost, data6Cyl_low.cumulativeCost, data8Cyl_low.cumulativeCost, data8Cyl_low.cumulativeCost, data8Cyl_low.cumulativeCost, data8Cyl_low.cumulativeCost, sales, 3)
	# this seems to work 

	# avg
	pi, cost = cleanData(data.PercentI4Avg.values, data.CostI4Avg.values*1.5, data.TechCat, data.Abbreviation)
	data4Cyl_avg = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)
	#print(data4Cyl_avg.dMPG)

	pi, cost = cleanData(data.PercentV6Avg.values, data.CostV6Avg.values*1.5, data.TechCat, data.Abbreviation)
	data6Cyl_avg = bcCalcOrder(pi/100, baselineMPG[1], cost, mpgLim)
	#print(data6Cyl_avg.dMPG)

	pi, cost = cleanData(data.PercentV8Avg.values, data.CostV8Avg.values*1.5, data.TechCat, data.Abbreviation)
	data8Cyl_avg = bcCalcOrder(pi/100, baselineMPG[2], cost, mpgLim)
	#print(data8Cyl_avg.dMPG)

	x1 = np.transpose(np.vstack((data4Cyl_avg.dMPG, data4Cyl_avg.dMPG2)))
	x2 = np.transpose(np.vstack((data6Cyl_avg.dMPG, data6Cyl_avg.dMPG2)))
	x3 = np.transpose(np.vstack((data8Cyl_avg.dMPG, data8Cyl_avg.dMPG2)))
	outData[1, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, data4Cyl_avg.cumulativeCost, data6Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, sales, 3)
	#outData[4, 2:6] = combinedCostFit(en1, en2, en3, data4Cyl_avg.cumulativeCost, data6Cyl_avg.cumulativeCost, data8Cyl_avg.cumulativeCost, truckSales)

	# high performance, low cost 
	pi, cost = cleanData(data.PercentI4High.values, data.CostI4Low.values*1.5, data.TechCat, data.Abbreviation)
	data4Cyl_high = bcCalcOrder(pi/100, baselineMPG[0], cost, mpgLim)

	pi, cost = cleanData(data.PercentV6High.values, data.CostV6Low.values*1.5, data.TechCat, data.Abbreviation)
	data6Cyl_high = bcCalcOrder(pi/100, baselineMPG[1], cost, mpgLim)

	pi, cost = cleanData(data.PercentV8High.values, data.CostV8Low.values*1.5, data.TechCat, data.Abbreviation)
	data8Cyl_high = bcCalcOrder(pi/100, baselineMPG[2], cost, mpgLim)

	x1 = np.transpose(np.vstack((data4Cyl_high.dMPG, data4Cyl_high.dMPG2)))
	x2 = np.transpose(np.vstack((data6Cyl_high.dMPG, data6Cyl_high.dMPG2)))
	x3 = np.transpose(np.vstack((data8Cyl_high.dMPG, data8Cyl_high.dMPG2)))
	outData[0, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, data4Cyl_high.cumulativeCost, data6Cyl_high.cumulativeCost, data8Cyl_high.cumulativeCost, data8Cyl_high.cumulativeCost, data8Cyl_high.cumulativeCost, data8Cyl_high.cumulativeCost, sales, 3)
	# this seems to work 
	return(outData)

def clean2017(pi, cost, techCat, rel):
	dropList = np.array([])
	df = pd.DataFrame(pi, columns = ['pi'])
	df['cost'] = cost
	df['techCat'] = techCat
	df['rel'] = rel
	df['bc'] = df.pi/df.cost
	
	for n in range(len(cost)):
		if np.isnan(pi[n]) or np.isnan(cost[n]) or techCat[n] == "AltFuels":
			dropList = np.append(dropList, n)
		if df.bc[n]>1 or df.bc[n]<0:
			df.bc[n] = 1
	df = df.drop(dropList)
	df = df.reset_index(drop = True)

	#g = df.index[df.groupby(['techCat', 'rel'])['bc'].idxmax()]
	#t = df.index[df.techCat == 'Transmission'].tolist()
	#for n in range(len(g)):
	#	if t[0] <= g[n] <= t[-1]:
	#		t.remove(g[n])
	#df = df.drop(t)
	#df = df.reset_index(drop = True)
	
	# remove any duplicates 
	df = df.drop_duplicates(subset = ['pi', 'cost'], keep = 'first')
	df = df.reset_index(drop = True)

	pi = np.asarray(df.pi)
	cost = np.asarray(df.cost)
	return(pi, cost)

def import2017(vehicleType, year): 
	if vehicleType == "cars":
		sales = [7723, 1540, 253]
		baselineMPG = [39.14, 28.93, 23.75] #[32, 27.5, 19.9]#[38.9, 28.9, 23.7] ## # # ## # #these are much higher than david's
	else: 
		sales = [2041, 3594, 1504] #4,6,8 cylinders
		baselineMPG = [33.41, 25.60, 22.31] #[32, 27.5, 19.9] #[33.3, 25.6, 22.3] #[32, 27.5, 19.9]  # #
	# prices in 2010 data? 
	data = pd.read_csv("201720202025.csv")
	outData = np.zeros((3,7))
	outData[:,1] = np.ones(3)*2010

	mpgLim = 5000
	if year == 2017:
		outData[:,0] = np.ones(3)*2017

		#pi, cost = cleanData(data.I4_Percent_High_17.values, data.I4_Cost_Low_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.I4_Percent_High_17.values, data.I4_Cost_Low_17.values*1.5, data.TechCat, data.Relative_to)
		#pi, cost = clean2017(data.I4_Percent_Avg_17.values, data.I4_Cost_Low_17.values*1.5, data.TechCat, data.Relative_to)
		I4_Low_17 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)
		#pi, cost = cleanData(data.I4_Percent_Avg_17.values, data.I4_Cost_Avg_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.I4_Percent_Avg_17.values, data.I4_Cost_Avg_17.values*1.5, data.TechCat, data.Relative_to)
		I4_Avg_17 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)
		#pi, cost = cleanData(data.I4_Percent_Low_17.values, data.I4_Cost_High_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.I4_Percent_Low_17.values, data.I4_Cost_High_17.values*1.5, data.TechCat, data.Relative_to)
		#pi, cost = clean2017(data.I4_Percent_Avg_17.values, data.I4_Cost_High_17.values*1.5, data.TechCat, data.Relative_to)
		I4_High_17 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)

		#pi, cost = cleanData(data.V6_Percent_High_17.values, data.V6_Cost_Low_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.V6_Percent_High_17.values, data.V6_Cost_Low_17.values*1.5, data.TechCat, data.Relative_to)
		#pi, cost = clean2017(data.V6_Percent_Avg_17.values, data.V6_Cost_Low_17.values*1.5, data.TechCat, data.Relative_to)
		V6_Low_17 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)
		#pi, cost = cleanData(data.V6_Percent_Avg_17.values, data.V6_Cost_Avg_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.V6_Percent_Avg_17.values, data.V6_Cost_Avg_17.values*1.5, data.TechCat, data.Relative_to)
		V6_Avg_17 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)
		#pi, cost = cleanData(data.V6_Percent_Low_17.values, data.V6_Cost_High_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.V6_Percent_Low_17.values, data.V6_Cost_High_17.values*1.5, data.TechCat, data.Relative_to)
		#pi, cost = clean2017(data.V6_Percent_Avg_17.values, data.V6_Cost_High_17.values*1.5, data.TechCat, data.Relative_to)
		V6_High_17 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)

		#pi, cost = cleanData(data.V8_Percent_High_17.values, data.V8_Cost_Low_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.V8_Percent_High_17.values, data.V8_Cost_Low_17.values*1.5, data.TechCat, data.Relative_to)
		#pi, cost = clean2017(data.V8_Percent_Avg_17.values, data.V8_Cost_Low_17.values*1.5, data.TechCat, data.Relative_to)
		V8_Low_17 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		#pi, cost = cleanData(data.V8_Percent_Avg_17.values, data.V8_Cost_Avg_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.V8_Percent_Avg_17.values, data.V8_Cost_Avg_17.values*1.5, data.TechCat, data.Relative_to)
		V8_Avg_17 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		#pi, cost = cleanData(data.V8_Percent_Low_17.values, data.V8_Cost_High_17.values*1.5, data.TechCat)
		pi, cost = clean2017(data.V8_Percent_Low_17.values, data.V8_Cost_High_17.values*1.5, data.TechCat, data.Relative_to)
		#pi, cost = clean2017(data.V8_Percent_Avg_17.values, data.V8_Cost_High_17.values*1.5, data.TechCat, data.Relative_to)
		V8_High_17 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		
		x1 = np.transpose(np.vstack((I4_Low_17.dMPG, I4_Low_17.dMPG2)))
		x2 = np.transpose(np.vstack((V6_Low_17.dMPG, V6_Low_17.dMPG2)))
		x3 = np.transpose(np.vstack((V8_Low_17.dMPG, V8_Low_17.dMPG2)))
		outData[0,2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_Low_17.cumulativeCost, V6_Low_17.cumulativeCost, V8_Low_17.cumulativeCost, V8_Low_17.cumulativeCost, V8_Low_17.cumulativeCost, V8_Low_17.cumulativeCost, sales, 3)
		
		x1 = np.transpose(np.vstack((I4_Avg_17.dMPG, I4_Avg_17.dMPG2)))
		x2 = np.transpose(np.vstack((V6_Avg_17.dMPG, V6_Avg_17.dMPG2)))
		x3 = np.transpose(np.vstack((V8_Avg_17.dMPG, V8_Avg_17.dMPG2)))
		outData[1, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_Avg_17.cumulativeCost, V6_Avg_17.cumulativeCost, V8_Avg_17.cumulativeCost, V8_Avg_17.cumulativeCost, V8_Avg_17.cumulativeCost, V8_Avg_17.cumulativeCost, sales, 3)
		
		x1 = np.transpose(np.vstack((I4_High_17.dMPG, I4_High_17.dMPG2)))
		x2 = np.transpose(np.vstack((V6_High_17.dMPG, V6_High_17.dMPG2)))
		s3 = np.transpose(np.vstack((V8_High_17.dMPG, V8_High_17.dMPG2)))
		outData[2, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_High_17.cumulativeCost, V6_High_17.cumulativeCost, V8_High_17.cumulativeCost, V8_High_17.cumulativeCost, V8_High_17.cumulativeCost, V8_High_17.cumulativeCost, sales, 3)
	elif year == 2020: 
		outData[:,0] = np.ones(3)*2020
		pi, cost = clean2017(data.I4_Percent_High_20_25.values, data.I4_Cost_Low_20.values*1.5, data.TechCat, data.Relative_to)
		I4_Low_20 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)
		pi, cost = clean2017(data.I4_Percent_Avg_20_25.values, data.I4_Cost_Avg_20.values*1.5, data.TechCat, data.Relative_to)
		I4_Avg_20 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)
		pi, cost = clean2017(data.I4_Percent_Low_20_25.values, data.I4_Cost_High_20.values*1.5, data.TechCat, data.Relative_to)
		I4_High_20 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)

		pi, cost = clean2017(data.V6_Percent_High_20_25.values, data.V6_Cost_Low_20.values*1.5, data.TechCat, data.Relative_to)
		V6_Low_20 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)
		pi, cost = clean2017(data.V6_Percent_Avg_20_25.values, data.V6_Cost_Avg_20.values*1.5, data.TechCat, data.Relative_to)
		V6_Avg_20 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)
		pi, cost = clean2017(data.V6_Percent_Low_20_25.values, data.V6_Cost_High_20.values*1.5, data.TechCat, data.Relative_to)
		V6_High_20 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)

		pi, cost = clean2017(data.V8_Percent_High_20_25.values, data.V8_Cost_Low_20.values*1.5, data.TechCat, data.Relative_to)
		V8_Low_20 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		pi, cost = clean2017(data.V8_Percent_Avg_20_25.values, data.V8_Cost_Avg_20.values*1.5, data.TechCat, data.Relative_to)
		V8_Avg_20 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		pi, cost = clean2017(data.V8_Percent_Low_20_25.values, data.V8_Cost_High_20.values*1.5, data.TechCat, data.Relative_to)
		V8_High_20 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		
		x1 = np.transpose(np.vstack((I4_Low_20.dMPG, I4_Low_20.dMPG2)))
		x2 = np.transpose(np.vstack((V6_Low_20.dMPG, V6_Low_20.dMPG2)))
		x3 = np.transpose(np.vstack((V8_Low_20.dMPG, V8_Low_20.dMPG2)))
		outData[0,2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_Low_20.cumulativeCost, V6_Low_20.cumulativeCost, V8_Low_20.cumulativeCost, V8_Low_20.cumulativeCost, V8_Low_20.cumulativeCost, V8_Low_20.cumulativeCost, sales, 3)
		
		x1 = np.transpose(np.vstack((I4_Avg_20.dMPG, I4_Avg_20.dMPG2)))
		x2 = np.transpose(np.vstack((V6_Avg_20.dMPG, V6_Avg_20.dMPG2)))
		x3 = np.transpose(np.vstack((V8_Avg_20.dMPG, V8_Avg_20.dMPG2)))
		outData[1, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_Avg_20.cumulativeCost, V6_Avg_20.cumulativeCost, V8_Avg_20.cumulativeCost, V8_Avg_20.cumulativeCost, V8_Avg_20.cumulativeCost, V8_Avg_20.cumulativeCost, sales, 3)
		
		x1 = np.transpose(np.vstack((I4_High_20.dMPG, I4_High_20.dMPG2)))
		x2 = np.transpose(np.vstack((V6_High_20.dMPG, V6_High_20.dMPG2)))
		x3 = np.transpose(np.vstack((V8_High_20.dMPG, V8_High_20.dMPG2)))
		outData[2, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_High_20.cumulativeCost, V6_High_20.cumulativeCost, V8_High_20.cumulativeCost, V8_High_20.cumulativeCost, V8_High_20.cumulativeCost, V8_High_20.cumulativeCost, sales, 3)
	else:
		outData[:,0] = np.ones(3)*2025
		pi, cost = clean2017(data.I4_Percent_High_20_25.values, data.I4_Cost_Low_25.values*1.5, data.TechCat, data.Relative_to)
		I4_Low_25 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)
		pi, cost = clean2017(data.I4_Percent_Avg_20_25.values, data.I4_Cost_Avg_25.values*1.5, data.TechCat, data.Relative_to)
		I4_Avg_25 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)
		pi, cost = clean2017(data.I4_Percent_Low_20_25.values, data.I4_Cost_High_25.values*1.5, data.TechCat, data.Relative_to)
		I4_High_25 = bcCalcOrder(pi/100, baselineMPG[0], cost, baselineMPG[0]*10)

		pi, cost = clean2017(data.V6_Percent_High_20_25.values, data.V6_Cost_Low_25.values*1.5, data.TechCat, data.Relative_to)
		V6_Low_25 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)
		pi, cost = clean2017(data.V6_Percent_Avg_20_25.values, data.V6_Cost_Avg_25.values*1.5, data.TechCat, data.Relative_to)
		V6_Avg_25 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)
		pi, cost = clean2017(data.V6_Percent_Low_20_25.values, data.V6_Cost_High_25.values*1.5, data.TechCat, data.Relative_to)
		V6_High_25 = bcCalcOrder(pi/100, baselineMPG[1], cost, baselineMPG[1]*10)

		pi, cost = clean2017(data.V8_Percent_High_20_25.values, data.V8_Cost_Low_25.values*1.5, data.TechCat, data.Relative_to)
		V8_Low_25 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		pi, cost = clean2017(data.V8_Percent_Avg_20_25.values, data.V8_Cost_Avg_25.values*1.5, data.TechCat, data.Relative_to)
		V8_Avg_25 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		pi, cost = clean2017(data.V8_Percent_Low_20_25.values, data.V8_Cost_High_25.values*1.5, data.TechCat, data.Relative_to)
		V8_High_25 = bcCalcOrder(pi/100, baselineMPG[2], cost, baselineMPG[2]*10)
		
		x1 = np.transpose(np.vstack((I4_Low_25.dMPG, I4_Low_25.dMPG2)))
		x2 = np.transpose(np.vstack((V6_Low_25.dMPG, V6_Low_25.dMPG2)))
		x3 = np.transpose(np.vstack((V8_Low_25.dMPG, V8_Low_25.dMPG2)))
		outData[0,2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_Low_25.cumulativeCost, V6_Low_25.cumulativeCost, V8_Low_25.cumulativeCost, V8_Low_25.cumulativeCost, V8_Low_25.cumulativeCost, V8_Low_25.cumulativeCost, sales, 3)
		
		x1 = np.transpose(np.vstack((I4_Avg_25.dMPG, I4_Avg_25.dMPG2)))
		x2 = np.transpose(np.vstack((V6_Avg_25.dMPG, V6_Avg_25.dMPG2)))
		x3 = np.transpose(np.vstack((V8_Avg_25.dMPG, V8_Avg_25.dMPG2)))
		outData[1, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_Avg_25.cumulativeCost, V6_Avg_25.cumulativeCost, V8_Avg_25.cumulativeCost, V8_Avg_25.cumulativeCost, V8_Avg_25.cumulativeCost, V8_Avg_25.cumulativeCost, sales, 3)
		
		x1 = np.transpose(np.vstack((I4_High_25.dMPG, I4_High_25.dMPG2)))
		x2 = np.transpose(np.vstack((V6_High_25.dMPG, V6_High_25.dMPG2)))
		x3 = np.transpose(np.vstack((V8_High_25.dMPG, V8_High_25.dMPG2)))
		outData[2, 2:7] = combinedCostFit(x1, x2, x3, x3, x3, x3, I4_High_25.cumulativeCost, V6_High_25.cumulativeCost, V8_High_25.cumulativeCost, V8_High_25.cumulativeCost, V8_High_25.cumulativeCost, V8_High_25.cumulativeCost, sales, 3)
	return outData

	# return data for 1990 estimates

def bcCalcOrder(percentImprovement, baselineMPG, costData, limit):
	outData = pd.DataFrame()
	outData['percentMPG'] = percentImprovement
	outData['cost'] = costData
	outData['deltaMPG'] = percentImprovement * baselineMPG
	outData['bc'] = np.zeros(len(percentImprovement))
	
	for n in range(len(percentImprovement)):
		if outData.cost[n] > 0:
			outData.bc[n] = outData.percentMPG[n]/costData[n]
		else:
			outData.cost[n] = 0
			outData.bc[n] = 1

	# sort data by benefit-cost ratio
	data = outData.sort_values(by = 'bc', ascending = False)
	data = data.reset_index(drop=True)
	data['cumulativeCost'] = np.zeros(len(percentImprovement))
	data['dMPG'] = np.zeros(len(percentImprovement))
	data['gpm'] = np.zeros(len(percentImprovement))
	for n in range(len(percentImprovement)):
		data.cumulativeCost[n] = sum(data.cost[0:n+1]) # seems to match
		if n == 0:
			data.gpm[n] = (1/baselineMPG)*(1-data.percentMPG[n])
			data.dMPG[n] = (1/data.gpm[n])-baselineMPG
		else:
			data.gpm[n] = data.gpm[n-1]*(1-data.percentMPG[n])
			data.dMPG[n] = (1/data.gpm[n])-baselineMPG
	data['dMPG2'] = data.dMPG**2

	dropList = []
	for n in range(len(data.dMPG)):
		if data.dMPG[n]>limit:
			dropList = np.append(dropList, n)
	data = data.drop(dropList)
	data = data.reset_index(drop = True)

	#print(data)
	outData = data[['cumulativeCost', 'dMPG', 'dMPG2', 'gpm', 'bc']]
	return(outData)

def combinedCostFit(ex1, ex2, ex3, ex4, ex5, ex6, en1, en2, en3, en4, en5, en6, carSales, segments):
	mod1 = sm.OLS(en1, ex1)
	yhat1 = mod1.fit()
	mod2 = sm.OLS(en2, ex2)
	yhat2 = mod2.fit()
	mod3 = sm.OLS(en3, ex3)
	yhat3 = mod3.fit()
	if segments < 4: 
		carSales = np.append(carSales, [0,0,0], axis=0)
	elif segments < 5:
		carSales = np.append(carSales, [0,0], axis=0)
	elif segments < 6:
		carSales = np.append(carSales, [0], axis=0)

	mod4 = sm.OLS(en3, ex3)
	mod5 = sm.OLS(en3, ex3)
	mod6 = sm.OLS(en3, ex3)
	if segments > 3: 
		mod4 = sm.OLS(en4, ex4)
		if segments > 4:
			mod5 = sm.OLS(en5, ex5)
			if segments > 5:
				mod6 = sm.OLS(en6, ex6)

	yhat4 = mod4.fit()
	yhat5 = mod5.fit() 
	yhat6 = mod6.fit()

	yhat = np.zeros((1000,2))
	for n in range(1000):
		b1 = np.random.normal(yhat1.params[0], yhat1.bse[0])*carSales[0]/sum(carSales)
		b2 = np.random.normal(yhat2.params[0], yhat2.bse[0])*carSales[1]/sum(carSales)
		b3 = np.random.normal(yhat3.params[0], yhat3.bse[0])*carSales[2]/sum(carSales)
		b4 = np.random.normal(yhat4.params[0], yhat4.bse[0])*carSales[3]/sum(carSales)
		b5 = np.random.normal(yhat5.params[0], yhat5.bse[0])*carSales[4]/sum(carSales)
		b6 = np.random.normal(yhat6.params[0], yhat6.bse[0])*carSales[5]/sum(carSales)
		
		c1 = np.random.normal(yhat1.params[1], yhat1.bse[1])*carSales[0]/sum(carSales)
		c2 = np.random.normal(yhat2.params[1], yhat2.bse[1])*carSales[1]/sum(carSales)
		c3 = np.random.normal(yhat3.params[1], yhat3.bse[1])*carSales[2]/sum(carSales)
		c4 = np.random.normal(yhat4.params[1], yhat4.bse[1])*carSales[3]/sum(carSales)
		c5 = np.random.normal(yhat5.params[1], yhat5.bse[1])*carSales[4]/sum(carSales)
		c6 = np.random.normal(yhat6.params[1], yhat6.bse[1])*carSales[5]/sum(carSales)

		yhat[n,0] = b1 + b2 + b3 + b4 + b5 + b6
		yhat[n,1] = c1 + c2 + c3 + c4 + c5 + c6
	#print(ex1[-1,0], ex2[-1,0], ex3[-1,0])
	avgMPGEnd = (ex1[-1,0]*carSales[0]+ex2[-1,0]*carSales[1]+ex3[-1,0]*carSales[2]+
		ex4[-1,0]*carSales[3]+ex5[-1,0]*carSales[4]+ex6[-1,0]*carSales[5])/sum(carSales)
	#print(sum(carSales))
	#print(avgMPGEnd)
	dataOut = np.hstack((np.average(yhat, axis = 0), stats.sem(yhat, axis=0), avgMPGEnd))
	return(dataOut)

def inflationConversion(curveCoeffs, baseYr):
	infData = importInflation()
	a = np.where(infData[:,0] == baseYr)
	for m in range(curveCoeffs.shape[2]):	
		for n in range(curveCoeffs.shape[0]):
			yr = curveCoeffs[n,1,m]
			p = np.where(infData[:,0] == yr)
			cf = infData[a,2]/infData[p,2]
			curveCoeffs[n,2:6,m] = curveCoeffs[n,2:6,m]*cf
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
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[6,:]
		elif yearlyCoeffs[n+1,0]< curveCoeffs[1,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[0,0], drawnCoeffs[1,0]), max(drawnCoeffs[0,0], drawnCoeffs[1,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[0,1], drawnCoeffs[1,1]), max(drawnCoeffs[0,1], drawnCoeffs[1,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[1,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[1,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[2,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[1,0], drawnCoeffs[2,0]), max(drawnCoeffs[1,0], drawnCoeffs[2,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[1,1], drawnCoeffs[2,1]), max(drawnCoeffs[1,1], drawnCoeffs[2,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[2,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[2,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[3,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[2,0], drawnCoeffs[3,0]), max(drawnCoeffs[2,0], drawnCoeffs[3,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[2,1], drawnCoeffs[3,1]), max(drawnCoeffs[2,1], drawnCoeffs[3,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[3,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[3,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[4,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[3,0], drawnCoeffs[4,0]), max(drawnCoeffs[3,0], drawnCoeffs[4,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[3,1], drawnCoeffs[4,1]), max(drawnCoeffs[3,1], drawnCoeffs[4,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[4,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[4,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[5,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[4,0], drawnCoeffs[5,0]), max(drawnCoeffs[4,0], drawnCoeffs[5,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[4,1], drawnCoeffs[5,1]), max(drawnCoeffs[4,1], drawnCoeffs[5,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[5,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[5,:]
		else:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(drawnCoeffs[5,0], drawnCoeffs[6,0]), max(drawnCoeffs[5,0], drawnCoeffs[6,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(drawnCoeffs[5,1], drawnCoeffs[6,1]), max(drawnCoeffs[5,1], drawnCoeffs[6,1]),1)
	return yearlyCoeffs

def linearInterpCoeffs(curveCoeffs):
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
			yearlyCoeffs[n+1,1] = drawnCoeffs[0,0]*deltat[1]/deltat[2] + drawnCoeffs[1,0]*deltat[0]/deltat[2] 
			yearlyCoeffs[n+1,2] = drawnCoeffs[0,1]*deltat[1]/deltat[2] + drawnCoeffs[1,1]*deltat[0]/deltat[2]
		elif yearlyCoeffs[n+1,0]==curveCoeffs[1,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[1,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[2,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[1,0], curveCoeffs[2,0]-yearlyCoeffs[n+1,0], curveCoeffs[2,0]-curveCoeffs[1,0]]
			yearlyCoeffs[n+1,1] = drawnCoeffs[1,0]*deltat[1]/deltat[2] + drawnCoeffs[2,0]*deltat[0]/deltat[2]
			yearlyCoeffs[n+1,2] = drawnCoeffs[1,1]*deltat[1]/deltat[2] + drawnCoeffs[2,1]*deltat[0]/deltat[2]
		elif yearlyCoeffs[n+1,0]==curveCoeffs[2,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[2,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[3,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[2,0], curveCoeffs[3,0]-yearlyCoeffs[n+1,0], curveCoeffs[3,0]-curveCoeffs[2,0]]
			yearlyCoeffs[n+1,1] = drawnCoeffs[2,0]*deltat[1]/deltat[2] + drawnCoeffs[3,0]*deltat[0]/deltat[2]
			yearlyCoeffs[n+1,2] = drawnCoeffs[2,1]*deltat[1]/deltat[2] + drawnCoeffs[3,1]*deltat[0]/deltat[2]
		elif yearlyCoeffs[n+1,0]==curveCoeffs[3,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[3,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[4,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[3,0], curveCoeffs[4,0]-yearlyCoeffs[n+1,0], curveCoeffs[4,0]-curveCoeffs[3,0]]
			yearlyCoeffs[n+1,1] = drawnCoeffs[3,0]*deltat[1]/deltat[2] + drawnCoeffs[4,0]*deltat[0]/deltat[2]
			yearlyCoeffs[n+1,2] = drawnCoeffs[3,1]*deltat[1]/deltat[2] + drawnCoeffs[4,1]*deltat[0]/deltat[2]
		elif yearlyCoeffs[n+1,0]==curveCoeffs[4,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[4,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[5,0]:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[4,0], curveCoeffs[5,0]-yearlyCoeffs[n+1,0], curveCoeffs[5,0]-curveCoeffs[4,0]]
			yearlyCoeffs[n+1,1] = drawnCoeffs[4,0]*deltat[1]/deltat[2] + drawnCoeffs[5,0]*deltat[0]/deltat[2]
			yearlyCoeffs[n+1,2] = drawnCoeffs[4,1]*deltat[1]/deltat[2] + drawnCoeffs[5,1]*deltat[0]/deltat[2]
		elif yearlyCoeffs[n+1,0]==curveCoeffs[5,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[5,:]
		else:
			deltat = [yearlyCoeffs[n+1,0]-curveCoeffs[5,0], curveCoeffs[6,0]-yearlyCoeffs[n+1,0], curveCoeffs[6,0]-curveCoeffs[5,0]]
			yearlyCoeffs[n+1,1] = drawnCoeffs[5,0]*deltat[1]/deltat[2] + drawnCoeffs[6,0]*deltat[0]/deltat[2]
			yearlyCoeffs[n+1,2] = drawnCoeffs[5,1]*deltat[1]/deltat[2] + drawnCoeffs[6,1]*deltat[0]/deltat[2]
	#print(yearlyCoeffs)
	return yearlyCoeffs

def interpWalk(curveCoeffs):
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
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[6,:]
		elif yearlyCoeffs[n+1,0]< curveCoeffs[1,0]:	
			yearlyCoeffs[n+1,1] = np.random.uniform(min(yearlyCoeffs[n,1], drawnCoeffs[1,0]), max(yearlyCoeffs[n,1], drawnCoeffs[1,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(yearlyCoeffs[n,2], drawnCoeffs[1,1]), max(yearlyCoeffs[n,2], drawnCoeffs[1,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[1,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[1,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[2,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(yearlyCoeffs[n,1], drawnCoeffs[2,0]), max(yearlyCoeffs[n,1], drawnCoeffs[2,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(yearlyCoeffs[n,2], drawnCoeffs[2,1]), max(yearlyCoeffs[n,2], drawnCoeffs[2,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[2,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[2,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[3,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(yearlyCoeffs[n,1], drawnCoeffs[3,0]), max(yearlyCoeffs[n,1], drawnCoeffs[3,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(yearlyCoeffs[n,2], drawnCoeffs[3,1]), max(yearlyCoeffs[n,2], drawnCoeffs[3,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[3,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[3,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[4,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(yearlyCoeffs[n,1], drawnCoeffs[4,0]), max(yearlyCoeffs[n,1], drawnCoeffs[4,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(yearlyCoeffs[n,2], drawnCoeffs[4,1]), max(yearlyCoeffs[n,2], drawnCoeffs[4,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[4,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[4,:]
		elif yearlyCoeffs[n+1,0]<curveCoeffs[5,0]:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(yearlyCoeffs[n,1], drawnCoeffs[5,0]), max(yearlyCoeffs[n,1], drawnCoeffs[5,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(yearlyCoeffs[n,2], drawnCoeffs[5,1]), max(yearlyCoeffs[n,2], drawnCoeffs[5,1]),1)
		elif yearlyCoeffs[n+1,0]==curveCoeffs[5,0]:
			yearlyCoeffs[n+1,1:3] = drawnCoeffs[5,:]
		else:
			yearlyCoeffs[n+1,1] = np.random.uniform(min(yearlyCoeffs[n,1], drawnCoeffs[6,0]), max(yearlyCoeffs[n,1], drawnCoeffs[6,0]),1)
			yearlyCoeffs[n+1,2] = np.random.uniform(min(yearlyCoeffs[n,2], drawnCoeffs[6,1]), max(yearlyCoeffs[n,2], drawnCoeffs[6,1]),1)
	return yearlyCoeffs

def applyLearningRate(mpgImprovementData, yearlyCoeffs):
	# apply only to years prior to 2025 
	yearlyCosts = np.ones((76, 5))  # ok 
	yearlyCosts[:,0] = 1975 + np.linspace(0,75,76) # ok 
	yearlyCosts[:,1] = mpgImprovementData
	costYrs = [1976, 1981, 1991, 2000, 2008, 2018]

	yearlyCosts[:,2] = yearlyCosts[:,1]*yearlyCoeffs[:,1] + yearlyCoeffs[:,2]*yearlyCosts[:,1]**2 #what the total cost would be 
	LR = 0.02
	yearlyCosts[0,3:5] = 0

	for n in range(len(yearlyCosts)-1):
		if yearlyCosts[n+1, 0] in costYrs:
			yearlyCosts[n+1, 3] = yearlyCosts[n+1,2]
		elif yearlyCosts[n,3] > 0:
			yearlyCosts[n+1,3] = max(0, yearlyCosts[n+1,2]-yearlyCosts[n,2])
		elif yearlyCosts[n,3] == 0 and n>0:
			yearlyCosts[n+1,3] = max(0, yearlyCosts[n+1,2]-yearlyCosts[n-1,2])
		elif n>1:
			if yearlyCosts[n,3] == 0 and yearlyCosts[n-1, 3] == 0:
				yearlyCosts[n+1,3] == max(0, yearlyCosts[n+1,2]-yearlyCosts[n-2,2])
		elif n>2:
			if yearlyCosts[n,3] == 0 and yearlyCosts[n-1, 3] == 0 and yearlyCosts[n-2, 3] == 0:
				yearlyCosts[n+1,3] == max(0, yearlyCosts[n+1,2]-yearlyCosts[n-3,2])
		elif n>3:
			if yearlyCosts[n,3] == 0 and yearlyCosts[n-1, 3] == 0 and yearlyCosts[n-2, 3] == 0 and yearlyCosts[n-3,3] == 0:
				yearlyCosts[n+1,3] == max(0, yearlyCosts[n+1,2]-yearlyCosts[n-4,2])
		elif n>4:
			if yearlyCosts[n,3] == 0 and yearlyCosts[n-1, 3] == 0 and yearlyCosts[n-2, 3] == 0 and yearlyCosts[n-3,3] == 0 and yearlyCosts[n-4,3] == 0:
				yearlyCosts[n+1,3] == yearlyCosts[n+1,2]
		
		yearlyCosts[n+1,4] = yearlyCosts[n,4]*(1-LR) + yearlyCosts[n+1,3]
	return(yearlyCosts)

def importCPIData(vehicleType):
	cpiData = pd.read_csv("CPIDataAnnual.csv")
	yrdata = cpiData['Year']
	allvehicleData = cpiData['New Vehicles-Deflated'].values
	if vehicleType == 'cars':
		specificVehicleData = cpiData['Old Index_Cars_2018Baseline_deflated'].values
	else: 
		specificVehicleData = cpiData['Old Index_Trucks_2018Baseline_deflated'].values
	#cpiData = pd.read_csv("CPINewCarPrices1950_2019.csv").values
	return(yrdata, allvehicleData, specificVehicleData)

def makeFigure(costOut, figName, vehicleType, baseYr):
	plotData = np.ones((76,10))
	plotData[:,0] = 1975 + np.linspace(0,75,76)
	for n in range(len(plotData)):
		plotData[n,1] = np.percentile(costOut[n,4,:,1], 50)
		plotData[n,2] = np.percentile(costOut[n,4,:,1], 2.5)
		plotData[n,3] = np.percentile(costOut[n,4,:,1], 97.5)

		plotData[n,4] = np.percentile(costOut[n,4,:,0], 50)
		plotData[n,5] = np.percentile(costOut[n,4,:,0], 2.5)
		plotData[n,6] = np.percentile(costOut[n,4,:,0], 97.5)

		plotData[n,7] = np.percentile(costOut[n,4,:,2], 50)
		plotData[n,8] = np.percentile(costOut[n,4,:,2], 2.5)
		plotData[n,9] = np.percentile(costOut[n,4,:,2], 97.5)

	colors = [[0.9, 0.8, 0], [0, 0.9, 0.9], [0.9, 0, 0.5]]
	plt.figure(figsize=(6,4))
	plt.plot(plotData[:,0], plotData[:,1], '-', color = colors[0])
	plt.text(plotData[43,0], plotData[43,1], 'Average', color = colors[0], FontSize=8)
	plt.fill_between(plotData[:,0], plotData[:,2], plotData[:,3], facecolor = colors[0], alpha = 0.5)

	plt.plot(plotData[:,0], plotData[:,4], '-', color = colors[1])
	plt.text(plotData[43,0], plotData[43,4], 'Low Cost, High /n Improvement', color = colors[1], FontSize=8)
	plt.fill_between(plotData[:,0], plotData[:,5], plotData[:,6], facecolor = colors[1], alpha = 0.5)

	plt.plot(plotData[:,0], plotData[:,7], '-', color = colors[2])
	plt.text(plotData[43,0], plotData[43,7], 'High Cost, Low % Improvement', color = colors[2], FontSize=8)
	plt.fill_between(plotData[:,0], plotData[:,8], plotData[:,9], facecolor = colors[2], alpha = 0.5)

	plt.xlim(1975,2018)
	plt.ylim(0, 4000)
	plt.ylabel('Cumulative Cost ['+str(baseYr)+'$]')
	plt.xlabel('Year')
	if vehicleType == "cars":
		cpiDatayr, cpiData1, cpiData2 = importCPIData(vehicleType)
		plt.title('Passenger Vehicles')
		#plt.plot([1970, 2018], [1729, 1729], '-', color = [0.5, 0.5, 0.5])
		#plt.text(1980, 1830, 'Previous 2018 estimate')
		plt.plot(cpiData1, cpiData1, '--k')
		#plt.plot(cpiData2[:,0], cpiData2[:,1], '--b')
		#print('cars', plotData[43, :])
	else:
		plt.title('Light Trucks')
		plt.plot([1970, 2018], [949,949], '--k')
		#print('trucks', plotData[43, :])
	return(plotData)
	plt.savefig(figName, dpi=300)
	plt.clf()

def makeCompFigure(c1, c2, vehicleType, baseYr):
	plotData1 = np.ones((76,10))
	plotData2 = np.ones((76,10))
	plotData1[:,0] = 1975 + np.linspace(0,75,76)
	plotData2[:,0] = 1975 + np.linspace(0,75,76)
	for n in range(len(plotData1)):
		plotData1[n,1] = np.percentile(c1[n,4,:,1], 50)
		plotData1[n,2] = np.percentile(c1[n,4,:,1], 2.5)
		plotData1[n,3] = np.percentile(c1[n,4,:,1], 97.5)

		plotData1[n,4] = np.percentile(c1[n,4,:,0], 50)
		plotData1[n,5] = np.percentile(c1[n,4,:,0], 2.5)
		plotData1[n,6] = np.percentile(c1[n,4,:,0], 97.5)

		plotData1[n,7] = np.percentile(c1[n,4,:,2], 50)
		plotData1[n,8] = np.percentile(c1[n,4,:,2], 2.5)
		plotData1[n,9] = np.percentile(c1[n,4,:,2], 97.5)

		plotData2[n,1] = np.percentile(c2[n,4,:,1], 50)
		plotData2[n,2] = np.percentile(c2[n,4,:,1], 2.5)
		plotData2[n,3] = np.percentile(c2[n,4,:,1], 97.5)

		plotData2[n,4] = np.percentile(c2[n,4,:,0], 50)
		plotData2[n,5] = np.percentile(c2[n,4,:,0], 2.5)
		plotData2[n,6] = np.percentile(c2[n,4,:,0], 97.5)

		plotData2[n,7] = np.percentile(c2[n,4,:,2], 50)
		plotData2[n,8] = np.percentile(c2[n,4,:,2], 2.5)
		plotData2[n,9] = np.percentile(c2[n,4,:,2], 97.5)

	colors = [[0.9, 0.8, 0], [0.2, 0.8, 0.8], [0.7, 0.2, 0.5]]

	fig = plt.figure(figsize=(6,3.5))
	ax1 = fig.add_subplot(1,1,1)
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.plot(plotData1[:,0], plotData1[:,1], '-', color = colors[0])
	plt.text(plotData1[43,0], plotData1[43,1], 'Avg.', FontSize = 8)
	plt.plot(plotData2[:,0], plotData2[:,1], '--', color = colors[0])
	plt.fill_between(plotData1[:,0], plotData1[:,2], plotData1[:,3], facecolor = colors[0], alpha = 0.5)

	plt.plot(plotData1[:,0], plotData1[:,4], '-', color = colors[1])
	plt.text(plotData1[43,0], plotData1[43,4], 'Low', FontSize = 8)
	plt.plot(plotData2[:,0], plotData2[:,4], '--', color = colors[1])
	plt.fill_between(plotData1[:,0], plotData1[:,5], plotData1[:,6], facecolor = colors[1], alpha = 0.5)

	plt.plot(plotData1[:,0], plotData1[:,7], '-', color = colors[2])
	plt.text(plotData1[43,0], plotData1[43,7], 'High', FontSize = 8)
	plt.plot(plotData2[:,0], plotData2[:,7], '--', color = colors[2])
	plt.fill_between(plotData1[:,0], plotData1[:,8], plotData1[:,9], facecolor = colors[2], alpha = 0.5)

	plt.xlim(1975,2018)
	plt.ylim(0, 3000)
	#plt.text(1997, 3500, 'Monte Carlo Simulations = '+str(numSims), HorizontalAlignment='center')
	plt.ylabel('Cumulative Cost ['+str(baseYr)+'$]', FontSize = 8)
	plt.xlabel('Year', FontSize = 8)
	plt.legend(('Uniform Distribution', 'Linear Interpolation'))

	if vehicleType == "cars":
		plt.title('Passenger Vehicles')
		#plt.plot([1970, 2018], [1729, 1729], '-', color = [0.5, 0.5, 0.5])
		#plt.text(1980, 1830, 'Previous 2018 Estimate')
		figName = 'compFigCars.png'
	else:
		plt.title('Light Trucks')
		#plt.plot([1970, 2018], [949,949], '--k')
		figName = 'compFigTrucks.png'
	plt.savefig(figName, dpi=300)

def plotCostCurves(curves, vehicleType, baseYr, low, avg, high, prev):
	deltaMPG = np.linspace(0,12,13)
	fig = plt.figure(figsize=(7,4))
	labels = ('1975*', '1980', '1990', '1999', '2007', '2017', '2020', '2025')

	if vehicleType == "cars":
		compData = pd.read_csv('passCarComp.csv').values
	else:
		compData = pd.read_csv('lightTruckComp.csv').values 

	legInfo = []
	for n in range(curves.shape[0]):
		ax = fig.add_subplot(2,4,n+1)
		ax.tick_params(axis='both', which='major', labelsize=8)
		#plt.plot([0,15], [300,300], '--k')
		#plt.plot([2,2], [0, 5000], '--k')
		if avg == 1: 
			plt.plot(deltaMPG, curves[n,2,1]*deltaMPG + curves[n,3,1]*deltaMPG*deltaMPG)
			legInfo.append('New Avg')
			figName = 'curves'+vehicleType+'avg.png'
			if prev == 1:
				plt.plot(deltaMPG, np.transpose(compData[n]), '--')
				legInfo.append('Prev Avg')
		if low == 1:
			plt.plot(deltaMPG, curves[n,2,0]*deltaMPG + curves[n,3,0]*deltaMPG*deltaMPG)
			legInfo.append('New Low')
			figName = 'curves'+vehicleType+'low.png'
			if prev == 1: 
				plt.plot(deltaMPG, np.transpose(compData[n+8]), '--')
				legInfo.append('Prev Low')
		if high == 1:
			plt.plot(deltaMPG, curves[n,2,2]*deltaMPG + curves[n,3,2]*deltaMPG*deltaMPG)
			legInfo.append('New High')
			figName = 'curves'+vehicleType+'high.png'
			if prev == 1: 
				plt.plot(deltaMPG, np.transpose(compData[n+16]), '--')
				legInfo.append('Prev High')
		if avg == 1 and low == 1 and high == 1:
			figName = 'curves'+vehicleType+'all.png'
		else:
			plt.legend(legInfo)

		plt.ylim(0, 1000) #5000)
		plt.title(labels[n], FontSize = 8)
		

	plt.text(-30, -210, 'MPG Improvement from Base Year', FontSize=8)
	plt.text(-57, 2000, 'Increase in Retail Price Equivalent [2015$]', FontSize=8, rotation = 90)
	plt.subplots_adjust(left = 0.09, right = 0.95, bottom = 0.1, top = 0.95, wspace = 0.3, hspace = 0.27)

	#plt.xlabel('Change in MPG')
	plt.savefig(figName, dpi = 300)

def makePaperFig(curves, costOut, vehicleType, baseYr):
	figName = ('placeholder.png')
	plotData = makeFigure(costOut, figName, vehicleType, baseYr)
	deltaMPG = np.linspace(0, curves[:,-1,1],100)

	colors = [[0.9, 0.8, 0], [0.2, 0.8, 0.8], [0.7, 0.2, 0.5]]

	fig = plt.figure(figsize=(7,4))
	ax1 = fig.add_subplot(3,3,7) # plot the average cost curves
	ax1.set_position([0.09, 0.15, 0.28, 0.8])
	labels = ('1975', '1980', '1992', '2002', '2007', '2017', '2020', '2025')

	for n in range(8):
		plt.plot(deltaMPG[:,n], curves[n,2,1]*deltaMPG[:,n] + curves[n,3,1]*deltaMPG[:,n]*deltaMPG[:,n])
		if n < 5:
			plt.text(deltaMPG[-1,n], curves[n,2,1]*deltaMPG[-1,n] + curves[n,3,1]*deltaMPG[-1,n]*deltaMPG[-1,n], labels[n], FontSize=8, HorizontalAlignment = 'right')
		else: 
			plt.text(deltaMPG[-1,n], curves[n,2,1]*deltaMPG[-1,n] + curves[n,3,1]*deltaMPG[-1,n]*deltaMPG[-1,n], labels[n], FontSize=8)
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.ylabel('Increase in Retail Price Equivalent ['+str(baseYr)+'$]', FontSize=8)
	plt.text(42, 9500, 'a', FontSize=8)
	plt.xlabel('MPG Improvement from Base Year', FontSize=8)
	plt.xlim(0,45)
	plt.ylim(0, 10000)
	
	doublex = 0.46
	xlen = 0.5
	ax2 = fig.add_subplot(3,3,2) # plot the cpi data
	ax2.set_position([doublex, 0.7, xlen, 0.25])
	cpiDatayr, cpiData1, cpiData2= importCPIData(vehicleType)
	ax2.plot(cpiDatayr, cpiData1, ':', color = [0, 0.5, 0.5]) #change this 
	ax2.plot(cpiDatayr, cpiData2, '-', color = [0.3, 0.3, 0.3]) #change colors 
	ax2.text(cpiDatayr[10], cpiData1[10]+5, 'New Vehicles', FontSize=8, color = [0, 0.5, 0.5])
	ax2.text(cpiDatayr[57], cpiData2[57]+15, 'Car Transactions', FontSize=8, color = [0.3, 0.3, 0.3], HorizontalAlignment = 'center')
	plt.ylabel('CPI', FontSize=8)
	plt.ylim([0, 150])
	plt.xlim(1950,2018)
	ax2.text(2013, 150*0.85, 'b', FontSize=8)
	ax2.tick_params(axis='both', which='major', labelsize=8)

	ax3 = fig.add_subplot(3,3,8) # plot the cumulative cost data
	ax3.set_position([doublex, 0.15, xlen, 0.45])
	ax3.tick_params(axis='both', which='major', labelsize=8)

	ax3.plot(plotData[:,0], plotData[:,1], '-', color = colors[0])
	ax3.text(plotData[43,0], plotData[43,1], 'Avg.', FontSize=8, color = colors[0])
	ax3.fill_between(plotData[:,0], plotData[:,2], plotData[:,3], facecolor = colors[0], alpha = 0.5)

	ax3.plot(plotData[:,0], plotData[:,4], '-', color = colors[1])
	ax3.text(plotData[43,0], plotData[43,4], 'Low', FontSize=8, color = colors[1])
	ax3.fill_between(plotData[:,0], plotData[:,5], plotData[:,6], facecolor = colors[1], alpha = 0.5)

	ax3.plot(plotData[:,0], plotData[:,7], '-', color = colors[2])
	ax3.text(plotData[43,0], plotData[43,7], 'High', FontSize=8, color = colors[2])
	ax3.fill_between(plotData[:,0], plotData[:,8], plotData[:,9], facecolor = colors[2], alpha = 0.5)

	plt.xlim(1950,2018)
	plt.ylim(0, 3000)
	plt.ylabel('Cumulative Cost ['+str(baseYr)+'$]', FontSize=8)
	plt.xlabel('Year', FontSize=8)
	ax3.text(2013, 3000*0.9, 'c', FontSize=8)

	print('cars', plotData[32, :])
	print('cars', plotData[43, :])
	
	plt.savefig('Figure4.png', dpi = 300)

def makeSIFig(curves, costOut, vehicleType, baseYr):
	figName = ('placeholder.png')
	plotData = makeFigure(costOut, figName, vehicleType, baseYr)
	deltaMPG = np.linspace(0, curves[:,-1,1],100)

	colors = [[0.9, 0.8, 0], [0.2, 0.8, 0.8], [0.7, 0.2, 0.5]]

	fig = plt.figure(figsize=(7,4))
	ax1 = fig.add_subplot(3,3,7) # plot the average cost curves
	ax1.set_position([0.09, 0.15, 0.28, 0.8])
	labels = ('1975', '1980', '1992', '2002', '2007', '2017', '2020', '2025')

	for n in range(8):
		plt.plot(deltaMPG[:,n], curves[n,2,1]*deltaMPG[:,n] + curves[n,3,1]*deltaMPG[:,n]*deltaMPG[:,n])
		if n < 5:
			plt.text(deltaMPG[-1,n], curves[n,2,1]*deltaMPG[-1,n] + curves[n,3,1]*deltaMPG[-1,n]*deltaMPG[-1,n], labels[n], FontSize=8, HorizontalAlignment = 'right')
		else: 
			plt.text(deltaMPG[-1,n], curves[n,2,1]*deltaMPG[-1,n] + curves[n,3,1]*deltaMPG[-1,n]*deltaMPG[-1,n], labels[n], FontSize=8)
	ax1.tick_params(axis='both', which='major', labelsize=8)
	plt.ylabel('Increase in Retail Price Equivalent [2015$]', FontSize=8)
	plt.text(42, 10500, 'a', FontSize=8)
	plt.xlabel('MPG Improvement from Base Year', FontSize=8)
	plt.xlim(0,45)
	plt.ylim(0, 11000)
	
	doublex = 0.46
	xlen = 0.5
	ax2 = fig.add_subplot(3,3,2) # plot the cpi data
	ax2.set_position([doublex, 0.7, xlen, 0.25])
	cpiDatayr, cpiData1, cpiData2= importCPIData(vehicleType)
	ax2.plot(cpiDatayr, cpiData1, ':', color = [0, 0.5, 0.5]) #change this 
	ax2.plot(cpiDatayr, cpiData2, '-', color = [0.3, 0.3, 0.3]) #change colors 
	ax2.text(cpiDatayr[10], cpiData1[10]+5, 'New Vehicles', FontSize=8, color = [0, 0.5, 0.5])
	ax2.text(cpiDatayr[57], cpiData2[57]+15, 'Truck Transactions', FontSize=8, color = [0.3, 0.3, 0.3], HorizontalAlignment='center')
	plt.ylabel('CPI', FontSize=8)
	plt.ylim([0, 150])
	plt.xlim(1950,2018)
	ax2.text(2013, 150*0.85, 'b', FontSize=8)
	ax2.tick_params(axis='both', which='major', labelsize=8)

	ax3 = fig.add_subplot(3,3,8) # plot the cumulative cost data
	ax3.set_position([doublex, 0.15, xlen, 0.45])

	#ax3 = fig.add_subplot(3,3,8) # plot the cumulative cost data
	#ax3.set_position([doublex, 0.15, xlen, 0.8])
	ax3.tick_params(axis='both', which='major', labelsize=8)

	ax3.plot(plotData[:,0], plotData[:,1], '-', color = colors[0])
	ax3.text(plotData[43,0], plotData[43,1], 'Avg.', FontSize=8, color = colors[0])
	ax3.fill_between(plotData[:,0], plotData[:,2], plotData[:,3], facecolor = colors[0], alpha = 0.5)

	ax3.plot(plotData[:,0], plotData[:,4], '-', color = colors[1])
	ax3.text(plotData[43,0], plotData[43,4], 'Low', FontSize=8, color = colors[1])
	ax3.fill_between(plotData[:,0], plotData[:,5], plotData[:,6], facecolor = colors[1], alpha = 0.5)

	ax3.plot(plotData[:,0], plotData[:,7], '-', color = colors[2])
	ax3.text(plotData[43,0], plotData[43,7], 'High', FontSize=8, color = colors[2])
	ax3.fill_between(plotData[:,0], plotData[:,8], plotData[:,9], facecolor = colors[2], alpha = 0.5)

	plt.xlim(1950,2018)
	plt.ylim(0, 3000)
	plt.ylabel('Cumulative Cost ['+str(baseYr)+'$]', FontSize=8)
	plt.xlabel('Year', FontSize=8)
	ax3.text(2013, 3000*0.9, 'c', FontSize=8)

	print('trucks', plotData[32, :])
	print('trucks', plotData[43, :])
	
	plt.savefig('TruckCumulativeCosts.png', dpi = 300)
