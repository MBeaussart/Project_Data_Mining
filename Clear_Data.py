import pandas as pd
import csv
import numpy as np
import time

#i import to much things...

####--------------------------------------------
#				clear beijing_17_18_aq training and test
####--------------------------------------------
def clear_beijing_17_18_aq_AND_201803():
	print("- beijing_17_18_aq.csv and beijing_201802_201803_aq", end=' ')
	#traing
	data_beijing_17_18_aq = pd.read_csv("final_project2018_data/beijing_17_18_aq.csv")
	#test
	data_beijing_201802_201803_aq = pd.read_csv("final_project2018_data/beijing_201802_201803_aq.csv")

	#define 'utc_time' column as time
	data_beijing_17_18_aq['utc_time'] = pd.to_datetime(data_beijing_17_18_aq['utc_time'])
	data_beijing_201802_201803_aq['utc_time'] = pd.to_datetime(data_beijing_201802_201803_aq['utc_time'])

	#remove row with no data (except stationId,utc_time) and duplicates
	data_beijing_17_18_aq = data_beijing_17_18_aq.dropna(thresh=3)
	data_beijing_17_18_aq = data_beijing_17_18_aq.drop_duplicates()
	data_beijing_201802_201803_aq = data_beijing_201802_201803_aq.dropna(thresh=3)
	data_beijing_201802_201803_aq = data_beijing_201802_201803_aq.drop_duplicates()

	data_beijing_17_18_aq.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_aq.csv', index=False)
	data_beijing_201802_201803_aq.to_csv('final_project2018_data/CSV_Create_Plot/beijing_201802_201803_aq.csv', index=False)
	print("[X]")



####--------------------------------------------
#				create data begining neural network
####--------------------------------------------
def clear_beijing_17_18_aq_AND_201803():
	print("- beijing_17_18_aq_deep", end=' ')

	data_beijing_17_18_aq_deep = pd.read_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_aq.csv')

	data_beijing_17_18_aq_deep = data_beijing_17_18_aq_deep.loc[lambda df: df['stationId']=="aotizhongxin_aq",:].fillna(0)
	data_beijing_17_18_aq_deep.drop(data_beijing_17_18_aq_deep.columns[0], axis=1, inplace=True)

	#define 'utc_time' column as time
	data_beijing_17_18_aq_deep['utc_time'] = pd.to_datetime(data_beijing_17_18_aq_deep['utc_time'])


	data_beijing_17_18_aq_deep.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_aq_deep.csv', index=False)
	print("[X]")





####--------------------------------------------
#				create Beijing_Weather_Stations_en
####--------------------------------------------
def create_Beijing_Weather_Stations_en():
	print("- Beijing_Weather_Stations_en.csv", end=" ")
	data_beijing_17_18_meo = pd.read_csv("final_project2018_data/beijing_17_18_meo.csv")
	data_beijing_17_18_meo=data_beijing_17_18_meo[["station_id","longitude","latitude"]].drop_duplicates()
	data_beijing_17_18_meo.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_Weather_Stations_en.csv', index=False)
	print("[X]")


####--------------------------------------------
#				clear beijing_17_18_meo training and test
####--------------------------------------------
def clear_beijing_17_18_meo_AND_201803():
	print("- beijing_17_18_meo.csv and beijing_201802_201803_me", end=" ")
	data_beijing_17_18_meo = pd.read_csv("final_project2018_data/beijing_17_18_meo.csv")
	data_beijing_201802_201803_me = pd.read_csv("final_project2018_data/beijing_201802_201803_me.csv")

	#delete columns with coordinate because we have the stationId
	data_beijing_17_18_meo.drop(data_beijing_17_18_meo.columns[[1,2]], axis=1, inplace=True)

	#define 'utc_time' column as time
	data_beijing_17_18_meo['utc_time'] = pd.to_datetime(data_beijing_17_18_meo['utc_time'])
	data_beijing_201802_201803_me['utc_time'] = pd.to_datetime(data_beijing_201802_201803_me['utc_time'])

	#remove row with no data (except stationId,utc_time) and duplicates
	data_beijing_17_18_meo = data_beijing_17_18_meo.dropna(thresh=3)
	data_beijing_17_18_meo = data_beijing_17_18_meo.drop_duplicates()
	data_beijing_201802_201803_me = data_beijing_201802_201803_me.dropna(thresh=3)
	data_beijing_201802_201803_me = data_beijing_201802_201803_me.drop_duplicates()

	#transforme weather to a number for classification
	i=0
	listWeather =['Sunny/clear', 'Overcast', 'Fog', 'Cloudy', 'Light Rain', 'Rain', 'Sleet', 'Snow', 'Thundershower', 'Hail', 'Rain with Hail', 'Rain/Snow with Hail', 'Haze', 'Dust', 'Sand']
	for idWeather in listWeather:
		data_beijing_17_18_meo.loc[data_beijing_17_18_meo.weather==idWeather, 'weather'] = i
		if str(idWeather) in np.array(data_beijing_201802_201803_me["weather"].unique()):
			data_beijing_201802_201803_me.loc[data_beijing_201802_201803_me.weather==idWeather, 'weather'] = i
		i=i+1

	data_beijing_17_18_meo.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_meo.csv', index=False)
	data_beijing_201802_201803_me.to_csv('final_project2018_data/CSV_Create_Plot/beijing_201802_201803_me.csv', index=False)
	print("[X]")


####--------------------------------------------
#				clear Beijing_AirQuality_Stations_en
####--------------------------------------------
def clear_Beijing_AirQuality_Stations_en():
	print("- Beijing_AirQuality_Stations_en.csv", end=" ")
	Beijing_AirQuality_Stations_en = pd.read_csv("final_project2018_data/Beijing_AirQuality_Stations_en.csv")
	Beijing_AirQuality_Stations_en.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_AirQuality_Stations_en.csv', index=False)
	print("[X]")

####--------------------------------------------
#				clear Beijing_grid_weather_station
####--------------------------------------------
def clear_Beijing_grid_weather_station():
	print("- Beijing_grid_weather_station.csv", end= " ")
	col_names= ['Station ID','latitude','longitude']
	Beijing_AirQuality_Stations_en = pd.read_csv("final_project2018_data/Beijing_grid_weather_station.csv",  names=col_names, header=None)
	Beijing_AirQuality_Stations_en.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_grid_weather_station.csv', index=False, columns=col_names)
	print("[X]")

####--------------------------------------------
#				clear Beijing_historical_meo_grid
####--------------------------------------------
def clear_Beijing_historical_meo_grid():
	print("- Beijing_historical_meo_grid.csv", end = " ", flush=True)
	data_Beijing_historical_meo_grid = pd.read_csv("final_project2018_data/Beijing_historical_meo_grid.csv")

	#delete columns coordinates because we have stationId
	data_Beijing_historical_meo_grid.drop(data_Beijing_historical_meo_grid.columns[[1,2]], axis=1, inplace=True)

	#define 'utc_time' column as time
	data_Beijing_historical_meo_grid['utc_time'] = pd.to_datetime(data_Beijing_historical_meo_grid['utc_time'])

	#remove row with no data (except stationId,utc_time) and duplicates
	data_Beijing_historical_meo_grid = data_Beijing_historical_meo_grid.dropna(thresh=3)
	data_Beijing_historical_meo_grid = data_Beijing_historical_meo_grid.drop_duplicates()

	data_Beijing_historical_meo_grid.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_historical_meo_grid.csv', index=False)
	print("[X]")


##----------------------------------------------
##create dataframe with column for each critere "O3, stationId .."and for each stationId and for each column and we have less row
##----------------------------------------------
# bug for now
def clear_beijing_17_18_columnByStation():
	print("- beijing_17_18_columnByStation.csv", end=" ")
	data_beijing_17_18_aq = pd.read_csv("final_project2018_data/CSV_Create_Plot/beijing_17_18_aq.csv")
	listStationId=data_beijing_17_18_aq["stationId"].unique()
	listWithStation={}
	i=0
	result=pd.DataFrame()

	for idOfStation in listStationId:
		listWithStation[i]=data_beijing_17_18_aq.loc[lambda df: df['stationId']==listStationId[i],:][['utc_time','PM2.5','PM10','NO2','CO','O3','SO2']]
		listWithStation[i]=listWithStation[i].add_suffix('_'+str(i))
		listWithStation[i]=listWithStation[i].reset_index(drop=True)
		result = pd.concat([result, listWithStation[i]], axis=1)
		i=i+1

	result.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_columnByStation.csv', index=False)
	print("[X]")

##----------------------------------------------
##create dataframe with column for each critere "O3, stationId .."and for each utc_time  and for each column and we have less row
##----------------------------------------------

def normalize(v):
	norm=np.linalg.norm(v, ord=1)
	if norm==0:
		norm=np.finfo(v.dtype).eps
	return v/norm

def clear_beijing_17_18_columnByDate():
	print("- beijing_17_18_columnByDate.csv")
	data_beijing_17_18_aq = pd.read_csv("final_project2018_data/CSV_Create_Plot/beijing_17_18_aq.csv")
	Dataframe_Date=data_beijing_17_18_aq["utc_time"].unique()
	tailleTotal=len(Dataframe_Date)
	listColumnnByDate={}
	i=0
	DataFrame_ColumnnByDate=pd.DataFrame()

	data_beijing_17_18_aq_normalize=data_beijing_17_18_aq.fillna(0)
	listColumn = list(data_beijing_17_18_aq_normalize.columns.values)
	for nameColumn in ['PM2.5','PM10','NO2','CO','O3','SO2']:
		data_beijing_17_18_aq_normalize[nameColumn]=normalize(data_beijing_17_18_aq_normalize[nameColumn])

	for idOfDate in Dataframe_Date:
		listColumnnByDate[i]=data_beijing_17_18_aq_normalize.loc[lambda df: df['utc_time']==Dataframe_Date[i],:][['PM2.5','PM10','NO2','CO','O3','SO2']]
		listColumnnByDate[i]=listColumnnByDate[i].add_suffix('_'+str(i))
		listColumnnByDate[i]=listColumnnByDate[i].reset_index(drop=True)
		DataFrame_ColumnnByDate=pd.concat([DataFrame_ColumnnByDate, listColumnnByDate[i]], axis=1)

		print("\t"+str(int(100*i/tailleTotal))+"%", end = "\r")
		i=i+1

	DataFrame_ColumnnByDate.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_columnByDate.csv', index=False)
	print("[X]")




#		main
#clear_beijing_17_18_aq_AND_201803()
clear_beijing_17_18_aq_AND_201803()
#create_Beijing_Weather_Stations_en()
#clear_beijing_17_18_meo_AND_201803()
#clear_Beijing_AirQuality_Stations_en()
#clear_Beijing_grid_weather_station()
#clear_Beijing_historical_meo_grid()
#clear_beijing_17_18_columnByStation()
#clear_beijing_17_18_columnByDate()
