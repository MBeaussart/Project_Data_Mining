import pandas as pd
import csv
import numpy as np
import time
import math


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
def clear_beijing_17_18_aq_deep():
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
	Beijing_grid_weather_station = pd.read_csv("final_project2018_data/Beijing_grid_weather_station.csv", names=col_names, header=None)
	Beijing_grid_weather_station.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_grid_weather_station.csv', index=False)
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

		print("\t"+str(int(100*i/tailleTotal)+1)+"%", end = "\r")
		i=i+1

	DataFrame_ColumnnByDate.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_columnByDate.csv', index=False)
	print("[X]")

##----------------------------------------------
##create dataframe station polution with column meteo
##----------------------------------------------
def giveRoundUP(x):
	return math.ceil(x*10)/10

def create_beijing_17_18_with_weather():
	print("- beijing_17_18_with_weather.csv", end=" ", flush=True)
	#data polution
	data_beijing_17_18_aq = pd.read_csv("final_project2018_data/CSV_Create_Plot/beijing_17_18_aq.csv")
	data_beijing_201802_201803_aq = pd.read_csv('final_project2018_data/CSV_Create_Plot/beijing_201802_201803_aq.csv')
	#data weather
	data_beijing_17_18_meo = pd.read_csv("final_project2018_data/CSV_Create_Plot/beijing_17_18_meo.csv")
	data_beijing_201802_201803_me = pd.read_csv("final_project2018_data/CSV_Create_Plot/beijing_201802_201803_me.csv")
	data_Beijing_historical_meo_grid = pd.read_csv("final_project2018_data/CSV_Create_Plot/Beijing_historical_meo_grid.csv")

	#data location station
	Beijing_grid_weather_station = pd.read_csv("final_project2018_data/CSV_Create_Plot/Beijing_grid_weather_station.csv")
	Beijing_AirQuality_Stations_en = pd.read_csv("final_project2018_data/CSV_Create_Plot/Beijing_AirQuality_Stations_en.csv")
	Beijing_Weather_Stations_en = pd.read_csv('final_project2018_data/CSV_Create_Plot/Beijing_Weather_Stations_en.csv')

	#define good columns
	Beijing_grid_weather_station.columns = ['stationId', 'latitude', 'longitude']
	Beijing_AirQuality_Stations_en.columns = ['stationId', 'longitude','latitude']
	Beijing_Weather_Stations_en.columns = ['stationId', 'longitude','latitude']

	#ADD some month of 2018 to data 17_18
	data_beijing_17_18_aq = data_beijing_17_18_aq.append(data_beijing_201802_201803_aq)
	data_beijing_17_18_meo = data_beijing_17_18_meo.append(data_beijing_201802_201803_me)
	#delete duplicates
	data_beijing_17_18_aq = data_beijing_17_18_aq.drop_duplicates()

	#group station weather
	Beijing_Weather_Stations_en = Beijing_Weather_Stations_en.append(Beijing_grid_weather_station)

	##	find for each polution station 4 weather stationId
	#add column longitude and latitude of station we want
	Beijing_AirQuality_Stations_en['longitude1'] = Beijing_AirQuality_Stations_en['longitude'].apply(giveRoundUP)+0.0
	Beijing_AirQuality_Stations_en['latitude1'] = Beijing_AirQuality_Stations_en['latitude'].apply(giveRoundUP)+0.0

	Beijing_AirQuality_Stations_en['longitude2'] = Beijing_AirQuality_Stations_en['longitude'].apply(giveRoundUP)
	Beijing_AirQuality_Stations_en['latitude2'] = Beijing_AirQuality_Stations_en['latitude'].apply(math.ceil)-0.1

	Beijing_AirQuality_Stations_en['longitude3'] = Beijing_AirQuality_Stations_en['longitude'].apply(giveRoundUP)-0.1
	Beijing_AirQuality_Stations_en['latitude3'] = Beijing_AirQuality_Stations_en['latitude'].apply(giveRoundUP)+0.0

	Beijing_AirQuality_Stations_en['longitude4'] = Beijing_AirQuality_Stations_en['longitude'].apply(giveRoundUP)-0.1
	Beijing_AirQuality_Stations_en['latitude4'] = Beijing_AirQuality_Stations_en['latitude'].apply(giveRoundUP)-0.1

	Beijing_AirQuality_Stations_en = Beijing_AirQuality_Stations_en.reset_index()
	data_beijing_17_18_aq = data_beijing_17_18_aq.reset_index()

	#add columns create to data_beijing_17_18_aq
	data_beijing_17_18_aq = pd.merge(data_beijing_17_18_aq, Beijing_AirQuality_Stations_en, on='stationId')
	data_beijing_17_18_aq.drop(["index_x","index_y"], axis=1, inplace=True)

	#find name weather station on the grid
	data_beijing_17_18_aq.drop(['longitude','latitude'], axis=1, inplace=True)
	data_beijing_17_18_aq['stationId_1']="beijing_grid_"+(( (data_beijing_17_18_aq['longitude1']-115)*210+(data_beijing_17_18_aq['latitude1']-39)*10)).apply(round).apply(str)
	data_beijing_17_18_aq.drop(['longitude1','latitude1'], axis=1, inplace=True)
	data_beijing_17_18_aq['stationId_2']="beijing_grid_"+(( (data_beijing_17_18_aq['longitude2']-115)*210+(data_beijing_17_18_aq['latitude2']-39)*10)).apply(round).apply(str)
	data_beijing_17_18_aq.drop(['longitude2','latitude2'], axis=1, inplace=True)
	data_beijing_17_18_aq['stationId_3']="beijing_grid_"+(( (data_beijing_17_18_aq['longitude3']-115)*210+(data_beijing_17_18_aq['latitude3']-39)*10)).apply(round).apply(str)
	data_beijing_17_18_aq.drop(['longitude3','latitude3'], axis=1, inplace=True)
	data_beijing_17_18_aq['stationId_4']="beijing_grid_"+(( (data_beijing_17_18_aq['longitude4']-115)*210+(data_beijing_17_18_aq['latitude4']-39)*10)).apply(round).apply(str)
	data_beijing_17_18_aq.drop(['longitude4','latitude4'], axis=1, inplace=True)

	#merge for each weather station next to the polution station, column about weather
	data_beijing_17_18_aq['stationName']=data_beijing_17_18_aq['stationId_1']
	data_beijing_17_18_aq.drop(['stationId_1'], axis=1, inplace=True)
	data_beijing_17_18_aq = pd.merge(data_beijing_17_18_aq, data_Beijing_historical_meo_grid, on=['utc_time','stationName'])

	#data_beijing_17_18_aq.drop(['stati'], axis=1, inplace=True)
	data_beijing_17_18_aq['stationName']=data_beijing_17_18_aq['stationId_2']
	data_beijing_17_18_aq.drop(['stationId_2'], axis=1, inplace=True)
	data_beijing_17_18_aq = pd.merge(data_beijing_17_18_aq, data_Beijing_historical_meo_grid, on=['utc_time','stationName'])

	#add weather station 1 and 2
	data_beijing_17_18_aq['temperature']=data_beijing_17_18_aq['temperature_x']+data_beijing_17_18_aq['temperature_y']
	data_beijing_17_18_aq.drop(['temperature_x','temperature_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['pressure']=data_beijing_17_18_aq['pressure_x']+data_beijing_17_18_aq['pressure_y']
	data_beijing_17_18_aq.drop(['pressure_x','pressure_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['humidity']=data_beijing_17_18_aq['humidity_x']+data_beijing_17_18_aq['humidity_y']
	data_beijing_17_18_aq.drop(['humidity_x','humidity_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['wind_direction']=data_beijing_17_18_aq['wind_direction_x']+data_beijing_17_18_aq['wind_direction_y']
	data_beijing_17_18_aq.drop(['wind_direction_x','wind_direction_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['wind_speed/kph']=data_beijing_17_18_aq['wind_speed/kph_x']+data_beijing_17_18_aq['wind_speed/kph_y']
	data_beijing_17_18_aq.drop(['wind_speed/kph_x','wind_speed/kph_y'], axis=1, inplace=True)

	#station 3
	data_beijing_17_18_aq['stationName']=data_beijing_17_18_aq['stationId_3']
	data_beijing_17_18_aq.drop(['stationId_3'], axis=1, inplace=True)
	data_beijing_17_18_aq = pd.merge(data_beijing_17_18_aq, data_Beijing_historical_meo_grid, on=['utc_time','stationName'])

	#add weather station 1&2 with 3
	data_beijing_17_18_aq['temperature']=data_beijing_17_18_aq['temperature_x']+data_beijing_17_18_aq['temperature_y']
	data_beijing_17_18_aq.drop(['temperature_x','temperature_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['pressure']=data_beijing_17_18_aq['pressure_x']+data_beijing_17_18_aq['pressure_y']
	data_beijing_17_18_aq.drop(['pressure_x','pressure_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['humidity']=data_beijing_17_18_aq['humidity_x']+data_beijing_17_18_aq['humidity_y']
	data_beijing_17_18_aq.drop(['humidity_x','humidity_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['wind_direction']=data_beijing_17_18_aq['wind_direction_x']+data_beijing_17_18_aq['wind_direction_y']
	data_beijing_17_18_aq.drop(['wind_direction_x','wind_direction_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['wind_speed/kph']=data_beijing_17_18_aq['wind_speed/kph_x']+data_beijing_17_18_aq['wind_speed/kph_y']
	data_beijing_17_18_aq.drop(['wind_speed/kph_x','wind_speed/kph_y'], axis=1, inplace=True)

	#station 4
	data_beijing_17_18_aq['stationName']=data_beijing_17_18_aq['stationId_4']
	data_beijing_17_18_aq.drop(['stationId_4'], axis=1, inplace=True)
	data_beijing_17_18_aq = pd.merge(data_beijing_17_18_aq, data_Beijing_historical_meo_grid, on=['utc_time','stationName'])
	data_beijing_17_18_aq.drop(['stationName'], axis=1, inplace=True)

	#add weather station 1&2&3 with 4
	data_beijing_17_18_aq['temperature']=(data_beijing_17_18_aq['temperature_x']+data_beijing_17_18_aq['temperature_y'])/4
	data_beijing_17_18_aq.drop(['temperature_x','temperature_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['pressure']=(data_beijing_17_18_aq['pressure_x']+data_beijing_17_18_aq['pressure_y'])/4
	data_beijing_17_18_aq.drop(['pressure_x','pressure_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['humidity']=(data_beijing_17_18_aq['humidity_x']+data_beijing_17_18_aq['humidity_y'])/4
	data_beijing_17_18_aq.drop(['humidity_x','humidity_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['wind_direction']=(data_beijing_17_18_aq['wind_direction_x']+data_beijing_17_18_aq['wind_direction_y'])/4
	data_beijing_17_18_aq.drop(['wind_direction_x','wind_direction_y'], axis=1, inplace=True)

	data_beijing_17_18_aq['wind_speed/kph']=(data_beijing_17_18_aq['wind_speed/kph_x']+data_beijing_17_18_aq['wind_speed/kph_y'])/4
	data_beijing_17_18_aq.drop(['wind_speed/kph_x','wind_speed/kph_y'], axis=1, inplace=True)
	data_beijing_17_18_aq = data_beijing_17_18_aq.sort_values(by=['stationId', 'utc_time'])
	data_beijing_17_18_aq.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_with_weather.csv', index=False)
	print("[X]")


#		main
#clear_beijing_17_18_aq_AND_201803()
#clear_beijing_17_18_aq_deep()
#create_Beijing_Weather_Stations_en()
#clear_beijing_17_18_meo_AND_201803()
#clear_Beijing_AirQuality_Stations_en()
#clear_Beijing_grid_weather_station()
#clear_Beijing_historical_meo_grid()
#clear_beijing_17_18_columnByStation()
#clear_beijing_17_18_columnByDate()
create_beijing_17_18_with_weather()
