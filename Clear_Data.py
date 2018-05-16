import pandas as pd
import csv
import numpy as np
import time
from itertools import combinations
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bokeh.layouts import row, widgetbox, column, gridplot, layout

from bokeh.models import HoverTool, ColumnDataSource, Select, Slider, CustomJS, DataRange1d, Plot, LinearAxis, Grid
from bokeh.models.markers import Circle
from bokeh.plotting import figure, show, output_file, ColumnDataSource, Figure
from bokeh.util.hex import hexbin

from bokeh.io import curdoc, show
from bokeh.transform import linear_cmap
#i import to much things...

####--------------------------------------------
#				clear beijing_17_18_aq training and test
####--------------------------------------------

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
print("- beijing_17_18_aq.csv and beijing_201802_201803_aq created and clean")


####--------------------------------------------
#				clear Beijing_AirQuality_Stations_en
####--------------------------------------------

Beijing_AirQuality_Stations_en = pd.read_csv("final_project2018_data/Beijing_AirQuality_Stations_en.csv")
Beijing_AirQuality_Stations_en.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_AirQuality_Stations_en.csv', index=False)
print("- Beijing_AirQuality_Stations_en.csv created and clean")


####--------------------------------------------
#				clear Beijing_grid_weather_station
####--------------------------------------------
col_names= ['Station ID','latitude','longitude']
Beijing_AirQuality_Stations_en = pd.read_csv("final_project2018_data/Beijing_grid_weather_station.csv",  names=col_names, header=None)
Beijing_AirQuality_Stations_en.to_csv('final_project2018_data/CSV_Create_Plot/Beijing_grid_weather_station.csv', index=False, columns=col_names)
print("- Beijing_grid_weather_station.csv created and clean")


##----------------------------------------------
##create dataframe with column for each critere "O3, stationId .."and for each stationId and for each column and we have less row
##----------------------------------------------
# bug for now
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
print("- beijing_17_18_columnByStation.csv created and clean")
##----------------------------------------------
##create dataframe with column for each critere "O3, stationId .."and for each utc_time  and for each column and we have less row
##----------------------------------------------

def normalize(v):
	norm=np.linalg.norm(v, ord=1)
	if norm==0:
		norm=np.finfo(v.dtype).eps
	return v/norm

Dataframe_Date=data_beijing_17_18_aq["utc_time"].unique()
print("wait until you have "+str(len(Dataframe_Date)))
time.sleep(5)
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
	print(i)
	i=i+1
DataFrame_ColumnnByDate.to_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_columnByDate.csv', index=False)
print("- beijing_17_18_columnByDate.csv created and clean")
