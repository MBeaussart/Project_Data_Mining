import pandas as pd
import csv
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bokeh.layouts import row, widgetbox, column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import figure, show, output_file, ColumnDataSource, Figure
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import HoverTool, ColumnDataSource, Select, Slider
from bokeh.plotting import figure


####-----------------------------------------------------------------------------
data_beijing_17_18_aq = pd.read_csv("final_project2018_data/beijing_17_18_aq.csv")
#define 'utc_time' column as time
data_beijing_17_18_aq['utc_time'] = pd.to_datetime(data_beijing_17_18_aq['utc_time'])

#   first of all clear useless data
#remove row with no data (except stationId,utc_time)
data_beijing_17_18_aq = data_beijing_17_18_aq.dropna(thresh=3)

#print(data_beijing_17_18_aq.isnull().sum())




###-------------------------------------------------------------------------------
#				plot
###-------------------------------------------------------------------------------



p = figure(title = "O3 evolution for a station ")
p.xaxis.axis_label = 'utc_time'
p.yaxis.axis_label = 'O3'
p.plot_width=1500

listStationId=data_beijing_17_18_aq["stationId"].unique()

listWithStation={}
i=0
result=pd.DataFrame()

#create dataframe with column for each critere "O3, stationId .." and for each stationId and we have less row because it's easy for the plot
for idOfStation in listStationId:
	listWithStation[i]=data_beijing_17_18_aq.loc[lambda df: df['stationId']==listStationId[i],:]
	listWithStation[i]=listWithStation[i].add_suffix('_'+str(i))
	listWithStation[i]=listWithStation[i].reset_index(drop=True)
	result = pd.concat([result, listWithStation[i]], axis=1)
	i=i+1
result=ColumnDataSource(result)

#for the first plot
source = ColumnDataSource(data_beijing_17_18_aq.loc[lambda df: df['stationId']==listStationId[0],:])
p.circle('utc_time', 'O3', source=source, fill_alpha=0.2, size=2)
output_file("plotPekin.html", title="plot pekin")


#instruction for slider
def callback(source=source, result=result, window=None):
	data = source.data
	f = cb_obj.value
	x, y = data['utc_time'], data['O3']

	data2=result.data
	x2, y2 = data2['utc_time_'+str(f)], data2['O3_'+str(f)]
	for i in range(len(x)):
		x[i] = x2[i]
		y[i] = y2[i]
	source.change.emit()
	source.data.update(source2.data)

id_slider = Slider(start=0, end=34, value=0, step=1, title="idStation",callback=CustomJS.from_py_func(callback))

layout = row(
	p,
	widgetbox(id_slider),
)

show(layout)
