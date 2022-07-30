from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import random

# unix, date, symbol, open, high, low, close, VolumeBTC, VolumeUSDT, tradecount
class DataPoint:
    def __init__(self, dataRow, indices):
        self.date = dataRow[indices[0]]
        self.open = float(dataRow[indices[1]])
        self.high = float(dataRow[indices[2]])
        self.low = float(dataRow[indices[3]])
        self.close = float(dataRow[indices[4]])

    def printDataPoint(self):
        print(f"Date: {self.date} open: {self.open} h: {self.high} l: {self.low} close: {self.close}")

def readData(filename, indices = [1,3,4,5,6]):
    data = []
    with open(filename, "r") as f:
        rawData = f.readlines()
        for row in reversed(rawData):
            if row[0] == '#':
                continue
            row = row.split(",")
            data.append(DataPoint(row, indices))
    return data


def plotData(data):
    x = range(len(data))
    prices = [o.open for o in data]
    plt.plot(x, prices)

class Simulation:
    def __init__(self, startMoney = 0):
        self.startMoney = startMoney
        self.money = self.startMoney
        self.stock = 0
        self.provision = 0.00

    def buy(self, amountOfMoney, price):
        if self.money >= amountOfMoney:
            self.money -= amountOfMoney
            self.stock += (1-self.provision)*amountOfMoney/price
    
    def sell(self, amountOfMoney, price):
        if self.stock >= amountOfMoney/price:
            self.money += (1-self.provision)*amountOfMoney
            self.stock -= amountOfMoney/price

    def totalValue(self, price):
        return self.money + self.stock * price


def allInStrategy(startBalance, x):
    return "BUY"

# Codzienie kupuje za x dolcow
def randomBuySellStrategy(startBalance, x):
    if random.randint(0, 1) == 1:
        return "BUY"
    else:
        return "SELL"
    
def average(data): 
    avg = sum(data) / len(data) 
    return avg

def movingAveragesStrategy(prices, x=0):
    avg50 = average(prices[len(prices)-50:])
    avg200 = average(prices[len(prices)-200:])
    if (avg50 > avg200):
        return "BUY"
    else:
        return "SELL"

def rsiStrategy(pricesSoFar, n):
    rsi = calculateRSI(pricesSoFar, 15)
    if (rsi > 80):
        return "SELL"
    elif (rsi < 20):
        return "BUY"
    else:
        return "HOLD"

def MMA(data):
    n = len(data)
    mma = data[0]
    for i in range(1, n):
        mma = ((n-1)*mma + data[i])/n
    return mma

def RS(data, n):
    relevantData = data[len(data) - n:len(data)]
    diffsInc = []
    diffsDec = []
    for i in range(1, len(relevantData)):
        diff = relevantData[i] - relevantData[i-1]
        if diff > 0:
            diffsInc.append(diff)
        elif diff < 0:
            diffsDec.append(-diff)
    return MMA(diffsInc)/MMA(diffsDec)

def calculateRSI(data, n = 15):
    return 100 - (100/(1 + RS(data, n)))


def simulate(startBalance, strategy, fractionOfTotalToTrade=1, step=1):
    totalValues = []
    moneyValues = []
    stockValues = []
    simulation = Simulation(startBalance)
    prices = [o.close for o in data]
    for i in range(15, len(prices), step):
        decision = strategy(prices[:i], 15)
        if (decision == "BUY"):
            simulation.buy(simulation.money * fractionOfTotalToTrade, data[i].close)
            # print(f"Buy. Curr stock: {simulation.stock} curr money: {simulation.money} for price {data[i].close}")
        elif (decision == "SELL"):
            simulation.sell(simulation.stock*fractionOfTotalToTrade*data[i].close, data[i].close)
            # print(f"Sell. Curr stock: {simulation.stock} curr money: {simulation.money} for price {data[i].close}")
        totalValues.append(simulation.totalValue(data[i].close))
        stockValues.append(simulation.stock * data[i].close)
        moneyValues.append(simulation.money)
    return [totalValues, moneyValues, stockValues]

def displayResultMsg(msg, totalValues, startBalance):
    if len(msg) == 0:
        msg = "Result"
    print(f"{msg}: Na koniec masz: {round(totalValues[-1])} dol z zainwestowanych {startBalance}"
            f" czyli {round((totalValues[-1]/startBalance) * 100)}%")

# Load data
data = readData("btc_every_h.csv")
# data = readData("HistoricalPrices.csv", [0,1,2,3,4])
data = data[::]
# data = data[int(len(data)*0.5):]

# Simulate
startBalance = 1000000
[totalValues, moneyValues, stockValues] = simulate(startBalance, movingAveragesStrategy, 0.1, 1)
displayResultMsg("movingAveragesStrategy", totalValues, startBalance)

# Display results
plt.figure()
plotData(data)

plt.figure()
plt.plot(range(len(data) - 15), totalValues, stockValues)

# Simulate all in
startBalance = 1000000
[totalValues, moneyValues, stockValues] = simulate(startBalance, allInStrategy, 1, 1)
displayResultMsg("allInStrategy", totalValues, startBalance)

# Display results

plt.figure()
plt.plot(range(len(data) - 15), totalValues, stockValues)


# plt.show()
