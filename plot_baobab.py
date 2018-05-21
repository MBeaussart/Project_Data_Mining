#run Deep.py before so you can have history.csv and plotAll.csv

import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot

data = pd.read_csv("history.csv")
pyplot.plot(data['col1'], label='train')
pyplot.plot(data['col2'], label='test')
pyplot.legend()
pyplot.show()


data = pd.read_csv("plotAll.csv")
pyplot.plot(data['col1'], label='train')
pyplot.plot(data['col2'], label='test')
pyplot.legend()
pyplot.show()

pyplot.plot(data['col3'], label='train')
pyplot.plot(data['col4'], label='test')
pyplot.legend()
pyplot.show()

pyplot.plot(data['col5'], label='train')
pyplot.plot(data['col6'], label='test')
pyplot.legend()
pyplot.show()

pyplot.plot(data['col7'], label='train')
pyplot.plot(data['col8'], label='test')
pyplot.legend()
pyplot.show()
