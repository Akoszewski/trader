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
            # print(f"Buy for price {price}")

    def sell(self, amountOfMoney, price):
        if self.stock >= amountOfMoney / price:
            self.money += (1 - self.provision) * amountOfMoney
            self.stock -= amountOfMoney / price
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

    def displayResultMsg(self, msg, lastValue, ratio):
        percentage = round(ratio * 100)
        if len(msg) == 0:
            msg = "Result"
        print(f"{msg}: Na koniec masz: {round(lastValue)} dol z zainwestowanych {self.startMoney}"
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
        if self.silent == False:
            self.displayResultMsg(strategy.__name__, lastValue, ratio)
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
    avg50 = average(pricesSoFar[len(pricesSoFar)-params[1]:])
    avg200 = average(pricesSoFar[len(pricesSoFar)-params[0]:])
    if (avg50 > avg200):
        return "BUY"
    else:
        return "SELL"

def rsiStrategy(pricesSoFar, params):
    rsi = calculateRSI(pricesSoFar, 15)
    if (rsi > 80):
        return "SELL"
    elif (rsi < 20):
        return "BUY"
    else:
        return "HOLD"

def RS(data, n):
    relevantData = data[len(data) - n:len(data)]
    diffsInc = []
    diffsDec = []
    for i in range(1, len(relevantData)):
        diff = relevantData[i] - relevantData[i-1]
        if diff >= 0:
            diffsInc.append(diff)
        elif diff <= 0:
            diffsDec.append(-diff)
    if len(diffsDec) > 0:
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

def getRandomDataChunk(minFractionOfLength, dataLength, delay, isChunkLengthFixed = False):
    diff = math.floor(minFractionOfLength * (dataLength - delay))
    left = random.randint(delay, dataLength - diff)
    if isChunkLengthFixed:
        right = left + diff
    else:
        right = random.randint(left + diff, dataLength)
    return [left, right]

def testStrategy(iterations, strategy, strategyParams, chunkSizeAsFraction, delay):
    ratios = []
    ratiosRef = []
    for i in range(iterations):
        [idxStart, idxEnd] = getRandomDataChunk(chunkSizeAsFraction, len(prices), delay, True)
        print(f"{i+1}/{iterations} (range {idxStart} - {idxEnd}):")

        simulation = Simulation(prices, idxStart, idxEnd, 1000000, 0.001, False)
        ratio = simulation.simulate(strategy, strategyParams, 1)
        ratios.append(ratio)

        ratioRef = simulation.simulate(holdStrategy, [], 1)
        ratiosRef.append(ratioRef)
        print("")

    averageRatio = sum(ratios)/len(ratios)
    print(f"Average ratio of start money for chosen strategy: {round(averageRatio * 100)}%")
    print(f"Average ratio of start money for holding: {round(sum(ratiosRef)/len(ratiosRef) * 100)}%")
    print(f"Best for chosen strategy: {round((max(ratios)) * 100)}%")
    print(f"Best for holding: {round((max(ratiosRef)) * 100)}%")
    print(f"Worst for chosen strategy: {round((min(ratios)) * 100)}%")
    print(f"Worst for holding: {round((min(ratiosRef)) * 100)}%")
    print("")
    return averageRatio

def combinedStrategy(pricesSoFar, params):
    smaDecision = movingAveragesStrategy(pricesSoFar, params)
    rsiDecision = rsiStrategy(pricesSoFar, params)
    if smaDecision == "BUY" and rsiDecision == "BUY":
        return "BUY"
    elif smaDecision == "SELL" and rsiDecision == "SELL":
        return "SELL"
    else:
        return "HOLD"

data = readData("btc.csv")
prices = [o.close for o in data]
# btcDataLen = 16487
# prices = prices[len(prices) - btcDataLen:]

# plt.plot(prices)
# plt.show()
chunkSizeAsFraction = 0.2
# modiffier1 = chunkSizeAsFraction * ratioToBtc * 50 # parameters to moving avg depend on planned time of trading
modiffier1 = 6
modiffier2 = modiffier1
delay = math.floor(200 * modiffier1) # delay must be at least the length of the data for calculating average
result = testStrategy(30, combinedStrategy, [delay, math.floor(50 * modiffier2)], chunkSizeAsFraction, delay)

# delay = math.floor(len(prices)*0.1) * 8
# [idxStart, idxEnd] = [delay, delay + math.floor(len(prices)*0.1)]
# simulation = Simulation(prices, idxStart, idxEnd, 1000000, 0.001)
# [ratio, isHoldingVect] = simulation.simulate(movingAveragesStrategy, [200 * 5, 50 * 5], 1)
# plt.plot(prices[idxStart:idxEnd])
# plt.plot(isHoldingVect)
# plt.show()

# Parameters tuning

# modiffs = []
# results = []
# iterations = 30
# for i in range(iterations):
#     print(f"Iteration: {i+1}/{iterations}")
#     modiffier1 = i + 1
#     modiffier2 = modiffier1
#     delay = 200 * modiffier1
#     result = testStrategy(10, movingAveragesStrategy, [delay, 50 * modiffier2], delay)
#     modiffs.append([modiffier1, modiffier2])
#     results.append(result)

# maxResult = max(results)
# maxIndex = results.index(maxResult)
# print(f"Best modiffs are: {modiffs[maxIndex]}")

# Two dim parameters tuning

# modiffier1 = 1
# modiffier2 = 1
# modiffs = []
# results = []
# iterations = 100
# for i in range(iterations):
#     print(f"Iteration: {i}/{iterations}")
#     modiffier1 = random.randint(1, 30)
#     modiffier2 = random.randint(1, 30)
#     delay = 200 * modiffier1
#     result = testStrategy(10, movingAveragesStrategy, [delay, 50 * modiffier2], delay)
#     modiffs.append([modiffier1, modiffier2])
#     results.append(result)

# maxResult = max(results)
# maxIndex = results.index(maxResult)
# print(f"Best modiffs are: {modiffs[maxIndex]}")

