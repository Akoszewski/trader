from datetime import date
import matplotlib.pyplot as plt
import numpy as np

# unix, date, symbol, open, high, low, close, VolumeBTC, VolumeUSDT, tradecount
class DataPoint:
    def __init__(self, dataRow):
        self.date = dataRow[1]
        self.open = float(dataRow[3])
        self.high = float(dataRow[4])
        self.low = float(dataRow[5])
        self.close = float(dataRow[6])

    def printDataPoint(self):
        print(f"Date: {self.date} open: {self.open} h: {self.high} l: {self.low} close: {self.close}")

def readData(filename):
    data = []
    with open(filename, "r") as f:
        rawData = f.readlines()
        for row in reversed(rawData):
            row = row.split(",")
            data.append(DataPoint(row))
    return data

def plotData(data):
    x = range(len(data))
    prices = [o.open for o in data]
    plt.plot(x, prices)
    plt.show()

data = readData("btc_every_h.csv")
plotData(data)