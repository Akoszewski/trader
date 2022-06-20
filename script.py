from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import random

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

class Simulator:
    def __init__(self, startMoney = 0):
        self.startMoney = startMoney
        self.money = self.startMoney
        self.crypto = 0

    def buy(self, amountOfMoney, price):
        if self.money >= amountOfMoney:
            self.money -= amountOfMoney
            self.crypto += amountOfMoney/price
    
    def sell(self, amountOfMoney, price):
        if self.crypto >= amountOfMoney/price:
            self.money += amountOfMoney
            self.crypto -= amountOfMoney/price

    def totalValue(self, price):
        return self.money + self.crypto * price

# Codzienie kupuje za x dolcow
def alwaysBuyStrategy(x):
    totalValues = []
    simulator = Simulator()
    for d in data:
        simulator.buy(x, d.open)
        totalValues.append(simulator.totalValue(d.open))
    return totalValues

# Codzienie kupuje za x dolcow
def alwaysBuyStrategy(startBalance, x):
    totalValues = []
    simulator = Simulator(startBalance)
    for d in data:
        simulator.buy(x, d.open)
        totalValues.append(simulator.totalValue(d.open))
    return totalValues

# Codzienie kupuje za x dolcow
def randomBuySellStrategy(startBalance, x):
    totalValues = []
    simulator = Simulator(startBalance)
    for d in data:
        choice = random.randint(0, 1)
        if choice == 1:
            simulator.buy(x, d.open)
        else:
            simulator.sell(x, d.open)
        totalValues.append(simulator.totalValue(d.open))
    return totalValues

data = readData("btc_every_day.csv")
# plotData(data)
startBalance = 1000000
totalValues = randomBuySellStrategy(startBalance, 100)
print(f"Na koniec masz: {round(totalValues[-1])} dol z zainwestowanych {startBalance}")

plt.plot(totalValues)
plt.show()
