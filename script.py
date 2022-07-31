from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import random
import math

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

class Account:
    def __init__(self, startMoney, provision):
        self.startMoney = startMoney
        self.money = self.startMoney
        self.stock = 0
        self.provision = provision

    def buy(self, amountOfMoney, price):
        if self.money >= amountOfMoney:
            self.money -= amountOfMoney
            self.stock += (1 - self.provision) * amountOfMoney / price

    def sell(self, amountOfMoney, price):
        if self.stock >= amountOfMoney / price:
            self.money += (1 - self.provision) * amountOfMoney
            self.stock -= amountOfMoney / price

    def totalValue(self, price):
        return self.money + self.stock * price

class Simulation:
    def __init__(self, data, delay, startMoney, provision = 0):
        self.prices = prices
        self.startMoney = startMoney
        self.provision = provision
        self.delay = delay

    def displayResultMsg(self, msg, totalValues, ratio):
        percentage = round(ratio * 100)
        if len(msg) == 0:
            msg = "Result"
        print(f"{msg}: Na koniec masz: {round(totalValues[-1])} dol z zainwestowanych {self.startMoney}"
                f" czyli {percentage}%")

    def simulate(self, strategy, fractionOfTotalToTrade=1, step=1):
        totalValues = []
        moneyValues = []
        stockValues = []
        account = Account(self.startMoney, self.provision)
        for i in range(self.delay, len(self.prices), step):
            decision = strategy(self.prices[:i], 15)
            if (decision == "BUY"):
                account.buy(account.money * fractionOfTotalToTrade, self.prices[i])
                # print(f"Buy. Curr stock: {self.stock} curr money: {self.money} for price {data[i].close}")
            elif (decision == "SELL"):
                account.sell(account.stock * fractionOfTotalToTrade * self.prices[i], self.prices[i])
                # print(f"Sell. Curr stock: {self.stock} curr money: {self.money} for price {data[i].close}")
            totalValues.append(account.totalValue(self.prices[i]))
            stockValues.append(account.stock * self.prices[i])
            moneyValues.append(account.money)
        ratio = totalValues[-1]/self.startMoney
        self.displayResultMsg(strategy.__name__, totalValues, ratio)
        return [totalValues, moneyValues, stockValues, ratio]

def holdStrategy(startBalance, x):
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


def getNumberedDataChunk(data, num, maxnum):
    chunkLen = math.floor(len(data) / maxnum)
    left = num * chunkLen
    right = num * chunkLen + chunkLen
    # print(left)
    # print(right)
    return data[left:right]


data = readData("btc_every_h.csv")
# data = readData("HistoricalPrices.csv", [0,1,2,3,4])
allPrices = [o.close for o in data]
# allPrices = allPrices[int(len(allPrices)*0.1):]

ratios = []
ratiosRef = []
maxchunk = 10
for i in range(maxchunk): 
    prices = getNumberedDataChunk(allPrices, i, maxchunk)
    simulation = Simulation(prices, 200, 1000000, 0.001)

    [totalValues, moneyValues, stockValues, ratio] = simulation.simulate(movingAveragesStrategy, 1)
    ratios.append(ratio)
    # plt.figure()
    # plt.plot(range(len(prices) - 15), totalValues, stockValues)

    [totalValues, moneyValues, stockValues, ratioRef] = simulation.simulate(holdStrategy, 1)
    ratiosRef.append(ratioRef)
    # plt.figure()
    # plt.plot(range(len(prices) - 15), totalValues, stockValues)
    print("")


print(f"Average ratio of start money for chosen strategy: {round(sum(ratios)/len(ratios) * 100)}%")
print(f"Average ratio of start money for holding: {round(sum(ratiosRef)/len(ratiosRef) * 100)}%")


plt.show()
