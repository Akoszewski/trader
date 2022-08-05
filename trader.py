from datetime import date
from pydoc import doc
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

class Account:
    def __init__(self, startMoney, provision):
        self.startMoney = startMoney
        self.money = self.startMoney
        self.stock = 0
        self.provision = provision
        self.trades = 0

    def buy(self, amountOfMoney, price):
        if self.money > 0:
            self.money -= amountOfMoney
            self.stock += (1 - self.provision) * amountOfMoney / price
            self.trades += 1
            # print(f"Buy for price {price}")

    def sell(self, amountOfMoney, price):
        if self.stock > 0:
            self.money += (1 - self.provision) * amountOfMoney
            self.stock -= amountOfMoney / price
            self.trades += 1
            # print(f"Sell for price {price}")

    def totalValue(self, price):
        return self.money + self.stock * price

class Simulation:
    def __init__(self, prices, idxStart, idxEnd, startMoney, provision = 0, silent = False):
        self.prices = prices
        self.startMoney = startMoney
        self.provision = provision
        self.idxStart = idxStart
        self.idxEnd = idxEnd
        self.silent = silent
        self.trades = 0

    def displayResultMsg(self, msg, lastValue, ratio, trades):
        percentage = round(ratio * 100)
        if len(msg) == 0:
            msg = "Result"
        print(f"{msg}: Na koniec po {trades} tranzakcjach masz: {round(lastValue)} dol z zainwestowanych {self.startMoney}"
                f" czyli {percentage}%")

    def simulate(self, strategy, strategyParams, fractionOfTotalToTrade=1, step=1):
        account = Account(self.startMoney, self.provision)
        lastValue = 0
        for i in range(self.idxStart, self.idxEnd, step):
            decision = strategy(self.prices[:i], strategyParams)
            if (decision == "BUY"):
                account.buy(account.money * fractionOfTotalToTrade, self.prices[i])
            elif (decision == "SELL"):
                account.sell(account.stock * fractionOfTotalToTrade * self.prices[i], self.prices[i])
            # totalValues.append(account.totalValue(self.prices[i]))
            lastValue = account.totalValue(self.prices[i])
        ratio = lastValue/self.startMoney
        self.trades = account.trades
        if self.silent == False:
            self.displayResultMsg(strategy.__name__, lastValue, ratio, self.trades)
        return ratio

def holdStrategy(pricesSoFar, params):
    return "BUY"

# Codzienie kupuje za x dolcow
def randomBuySellStrategy(pricesSoFar, params):
    if random.randint(0, 1) == 1:
        return "BUY"
    else:
        return "SELL"
    
def average(data):
    return sum(data) / len(data) 

def movingAveragesStrategy(pricesSoFar, params):
    avgShorter = average(pricesSoFar[len(pricesSoFar)-params[1]:])
    avgLonger = average(pricesSoFar[len(pricesSoFar)-params[0]:])
    if (avgShorter > avgLonger):
        return "BUY"
    else:
        return "SELL"

def rsiStrategy(pricesSoFar, params):
    rsi = calculateRSI(pricesSoFar, params[0])
    if (rsi > params[2]):
        return "SELL"
    elif (rsi < params[1]):
        return "BUY"
    else:
        return "HOLD"

def RS(data, n):
    relevantData = data[len(data) - n:]
    diffsInc = []
    diffsDec = []
    for i in range(1, len(relevantData)):
        diff = relevantData[i] - relevantData[i-1]
        if diff >= 0:
            diffsInc.append(diff)
        elif diff <= 0:
            diffsDec.append(-diff)
    if len(diffsDec) and len(diffsInc) > 0:
        return average(diffsInc)/average(diffsDec)
    else:
        return 1

def calculateRSI(data, n):
    return 100 - (100/(1 + RS(data, n)))


def getNumberedDataChunk(tradingDataLen, delay, num, maxnum):
    chunkLen = math.floor(tradingDataLen / maxnum)
    left = num * chunkLen + delay
    right = num * chunkLen + chunkLen + delay
    return [left, right]

def getRandomDataChunk(minChunkLength, dataLength, delay, isChunkLengthFixed = False):
    diff = minChunkLength
    left = random.randint(delay, dataLength - diff)
    if isChunkLengthFixed:
        right = left + diff
    else:
        right = random.randint(left + diff, dataLength)
    return [left, right]


def daysToIntervals(days, intervalsPerDay = 24):
    return math.floor(days * intervalsPerDay)

def combinedStrategy(pricesSoFar, params):
    smaDecision = movingAveragesStrategy(pricesSoFar, params)
    rsiDecision = rsiStrategy(pricesSoFar, params[2:])
    if smaDecision == "BUY" and rsiDecision == "BUY":
        return "BUY"
    elif smaDecision == "SELL" and rsiDecision == "SELL":
        return "SELL"
    else:
        return "HOLD"


class StrategyTester:
    def doSimulation(i, iterations, strategy, strategyParams, chunkSize, startDelay):
        [idxStart, idxEnd] = getRandomDataChunk(chunkSize, len(prices), startDelay, True)
        print(f"{i+1}/{iterations} (range {idxStart} - {idxEnd}):")

        simulation = Simulation(prices, idxStart, idxEnd, 1000000, 0.001, False)
        ratio = simulation.simulate(strategy, strategyParams, 1)

        ratioRef = simulation.simulate(holdStrategy, [], 1)
        print("")
        return [ratio, ratioRef]

    def testStrategy(iterations, strategy, strategyParams, chunkSize, startDelay):
        ratios = []
        ratiosRef = []
        for i in range(iterations):
            [ratio, ratioRef] = StrategyTester.doSimulation(i, iterations, strategy, strategyParams, chunkSize, startDelay)
            ratios.append(ratio)
            ratiosRef.append(ratioRef)

        averageRatio = sum(ratios)/len(ratios)
        print(f"Average ratio of start money for chosen strategy: {round(averageRatio * 100)}%")
        print(f"Average ratio of start money for holding: {round(sum(ratiosRef)/len(ratiosRef) * 100)}%")
        print(f"Best for chosen strategy: {round((max(ratios)) * 100)}%")
        print(f"Best for holding: {round((max(ratiosRef)) * 100)}%")
        print(f"Worst for chosen strategy: {round((min(ratios)) * 100)}%")
        print(f"Worst for holding: {round((min(ratiosRef)) * 100)}%")
        print("")
        return averageRatio



data = readData("./data/hourly/btc.csv")
prices = [o.close for o in data]

# plt.plot(prices)
# plt.show()

chunkSize = daysToIntervals(30)

smaParam1 = daysToIntervals(50)
smaParam2 = daysToIntervals(12)
[rsiParam1, rsiParam2, rsiParam3] = [daysToIntervals(0.6), 20, 80]
startDelay = daysToIntervals(365)
# startDelay = max([smaParam1, smaParam2, rsiParam1]) # delay must be at least the length of the data the decision is based on
combinedStrategyParams = [smaParam1, smaParam2, rsiParam1, rsiParam2, rsiParam3]
result = StrategyTester.testStrategy(100, combinedStrategy, combinedStrategyParams, chunkSize, startDelay)

