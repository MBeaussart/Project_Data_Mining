import pandas as pd
import csv
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from bokeh.layouts import row, widgetbox, column, gridplot, layout

from bokeh.models import HoverTool, ColumnDataSource, Select, Slider, CustomJS, DataRange1d, Plot, LinearAxis, Grid
from bokeh.models.markers import Circle
from bokeh.plotting import figure, show, output_file, ColumnDataSource, Figure
from bokeh.util.hex import hexbin

from bokeh.io import curdoc, show
from bokeh.transform import linear_cmap
#i import to much things...

####-----------------------------------------------------------------------------
#				data use
###-------------------------------------------------------------------------------
data_beijing_17_18_aq = pd.read_csv("final_project2018_data/CSV_Create_Plot/beijing_17_18_aq.csv")
Beijing_AirQuality_Stations_en = pd.read_csv("final_project2018_data/CSV_Create_Plot/Beijing_AirQuality_Stations_en.csv")
DataFrame_ColumnnByDate = pd.read_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_columnByDate.csv')
Beijing_grid_weather_station = pd.read_csv('final_project2018_data/CSV_Create_Plot/Beijing_grid_weather_station.csv')
#bug just down
#DataFrame_ColumnnByStation = pd.read_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_columnByStation.csv')

#define 'utc_time' column as time
data_beijing_17_18_aq['utc_time'] = pd.to_datetime(data_beijing_17_18_aq['utc_time'])


###-------------------------------------------------------------------------------
#				plot
###-------------------------------------------------------------------------------
plot_grid = Plot(title=None, plot_width=900, plot_height=900, h_symmetry=False, v_symmetry=False, min_border=0, toolbar_location=None)

plot_O3 = figure(title = "O3 evolution for a station ")
plot_CO = figure(title = "CO evolution for a station ")
plot_grid = figure(title = "Grid evolution of CO ")
plot_grid_station = figure(title = "Grid station ")

plot_O3.xaxis.axis_label = 'utc_time'
plot_CO.xaxis.axis_label = 'utc_time'
plot_grid.xaxis.axis_label = 'latitude'
plot_grid_station.xaxis.axis_label = 'latitude'

plot_O3.yaxis.axis_label = 'O3'
plot_CO.yaxis.axis_label = 'CO'
plot_grid.yaxis.axis_label = 'longitude'
plot_grid_station.yaxis.axis_label = 'longitude'

plot_O3.plot_width=1500
plot_CO.plot_width=1500
plot_grid.plot_width=900
plot_grid_station.plot_width=900

plot_grid.plot_height=900
plot_grid_station.plot_height=900


###-------------------------------------------------------------------------------
#				plots 1 (O3) and 2 (CO)
###-------------------------------------------------------------------------------

listStationId=data_beijing_17_18_aq["stationId"].unique()
listWithStation={}
i=0
result=pd.DataFrame()

#create dataframe with column for each critere "O3, stationId .." and for each stationId and we have less row because it's easy for the plot
for idOfStation in listStationId:
	listWithStation[i]=data_beijing_17_18_aq.loc[lambda df: df['stationId']==listStationId[i],:][['utc_time','PM2.5','PM10','NO2','CO','O3','SO2']]
	listWithStation[i]=listWithStation[i].add_suffix('_'+str(i))
	listWithStation[i]=listWithStation[i].reset_index(drop=True)
	result = pd.concat([result, listWithStation[i]], axis=1)
	i=i+1

result=ColumnDataSource(result)

output_file("plotPekin.html", title="plot pekin")

#the first plot (O3)
source = ColumnDataSource(data_beijing_17_18_aq.loc[lambda df: df['stationId']==listStationId[0],:])
plot_O3.circle('utc_time', 'O3', source=source, fill_alpha=0.2, size=2)


#the 2 plot (CO)
source_CO = ColumnDataSource(data_beijing_17_18_aq.loc[lambda df: df['stationId']==listStationId[0],:])
plot_CO.circle('utc_time', 'CO', source=source_CO, fill_alpha=0.2, size=2)

#instruction for slider O3
def callback_O3(source=source, result=result, window=None):
	data = source.data
	f = cb_obj.value
	x, y = data['utc_time'], data['O3']

	data2=result.data
	x2, y2 = data2['utc_time_'+str(f)], data2['O3_'+str(f)]
	for i in range(len(x)):
		x[i] = x2[i]
		y[i] = y2[i]
	source.change.emit()

slider_O3 = Slider(start=0, end=34, value=0, step=1, title="idStation",callback=CustomJS.from_py_func(callback_O3))

#instruction for slider CO
def callback_CO(source_CO=source_CO, result=result, window=None):
	data = source_CO.data
	f = cb_obj.value
	x, y = data['utc_time'], data['CO']

	data2=result.data
	x2, y2 = data2['utc_time_'+str(f)], data2['CO_'+str(f)]
	for i in range(len(x)):
		x[i] = x2[i]
		y[i] = y2[i]
	source_CO.change.emit()

slider_CO = Slider(start=0, end=34, value=0, step=1, title="idStation",callback=CustomJS.from_py_func(callback_CO))

##----------------------------------------------
#			grid plot
##----------------------------------------------

def normalize(v):
	norm=np.linalg.norm(v, ord=1)
	if norm==0:
		norm=np.finfo(v.dtype).eps
	return v/norm

Dataframe_Date=data_beijing_17_18_aq["utc_time"].unique()
DataFrame_ColumnnByDate=ColumnDataSource(DataFrame_ColumnnByDate)
longitude = Beijing_AirQuality_Stations_en['longitude']
latitude = Beijing_AirQuality_Stations_en['latitude']
sizes = DataFrame_ColumnnByDate.data['CO_0']*1000000*3
source_grid = ColumnDataSource(dict(longitude=longitude, latitude=latitude, sizes=sizes))

plot_grid.circle('longitude', 'latitude', source=source_grid, size="sizes", line_color="#307CDF", fill_color="#71B9EF", line_width=3)

#instruction for slider matrice
def callback_grid(source_grid=source_grid, DataFrame_ColumnnByDate=DataFrame_ColumnnByDate, window=None):
	data = source_grid.data
	f = cb_obj.value
	x,y = data['longitude'], data['latitude']
	taille = data['sizes']

	data2=DataFrame_ColumnnByDate.data
	taille2 = data2['CO_'+str(f)]
	for i in range(len(x)):
		x[i] = x[i]
		y[i] = y[i]
		taille[i] = taille2[i]*1000000*3
	source_grid.change.emit()

slider_grid = Slider(start=0, end=len(Dataframe_Date), value=0, width=800, step=1, title="Time",callback=CustomJS.from_py_func(callback_grid))


###-------------------------------------------------------------------------------
#				plots grid station
###-------------------------------------------------------------------------------


data_beijing_grid=pd.DataFrame()
data_beijing_grid = Beijing_AirQuality_Stations_en.append(Beijing_grid_weather_station)

#the first plot (O3)
source = ColumnDataSource(data_beijing_grid)
plot_grid_station.circle('latitude', 'longitude', source=source, fill_alpha=0.2, size=4)


layout = gridplot([[plot_O3, widgetbox(slider_O3)], [plot_CO, widgetbox(slider_CO)],[plot_grid, widgetbox(slider_grid)],[plot_grid_station]])
show(layout)

###-------------------------------------------------------------------------------
# 				matlab plot (example)
###-------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
					   linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
