from datetime import date
import csv

# unix, date, symbol, open, high, low, close, VolumeBTC, VolumeUSDT, tradecount
class DataPoint:
    def __init__(self, dataRow):
        self.date = dataRow[1]
        self.open = dataRow[3]
        self.high = dataRow[4]
        self.low = dataRow[5]
        self.close = dataRow[6]

    def printDataPoint(self):
        print(f"Date: {self.date} open: {self.open} h: {self.high} l: {self.low} close: {self.close}")

data = []
with open("btc_every_h.csv", "r") as f:
    rawData = f.readlines()
    for row in rawData:
        row = row.split(",")
        data.append(DataPoint(row))

data[0].printDataPoint()
