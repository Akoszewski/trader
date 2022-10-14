from datetime import date
from pydoc import doc
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
import pandas_ta as pta

# unix, date, symbol, open, high, low, close, VolumeBTC, VolumeUSDT, tradecount
class DataPoints:
    def __init__(self, rawData, indices):
        self.dates = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        for row in reversed(rawData):
            if row[0] == '#':
                continue
            row = row.split(",")
            self.dates.append(row[indices[0]])
            self.opens.append(float(row[indices[1]]))
            self.highs.append(float(row[indices[2]]))
            self.lows.append(float(row[indices[3]]))
            self.closes.append(float(row[indices[4]]))
            self.sma20 = []
            self.sma50 = []
            self.sma100 = []
            self.sma200 = []
            self.ema20 = []
            self.ema50 = []
            self.ema100 = []
            self.ema200 = []
            self.rsi = []
            self.macd = []

    # def printDataPoint(self):
        # print(f"Date: {self.date} open: {self.open} h: {self.high} l: {self.low} close: {self.close}")
    
    def initTechnicals(self):
        print(f"Calculating technicals...")
        prices = self.closes
        df = pd.DataFrame(prices, columns =['closes'])

        self.sma20 = pta.sma(close = df['closes'], length = 20).to_numpy()
        self.sma50 = pta.sma(close = df['closes'], length = 50).to_numpy()
        self.sma100 = pta.sma(close = df['closes'], length = 100).to_numpy()
        self.sma200 = pta.sma(close = df['closes'], length = 200).to_numpy()
        self.ema20 = pta.ema(close = df['closes'], length = 20).to_numpy()
        self.ema50 = pta.ema(close = df['closes'], length = 50).to_numpy()
        self.ema100 = pta.ema(close = df['closes'], length = 100).to_numpy()
        self.ema200 = pta.ema(close = df['closes'], length = 200).to_numpy()
        self.rsi = pta.rsi(close = df['closes'], length = 14).to_numpy()
        self.macd = pta.macd(close = df['closes'], length = 14).to_numpy()

def readData(filename, indices = [1,3,4,5,6]):
    data = []
    with open(filename, "r") as f:
        rawData = f.readlines()
        data = DataPoints(rawData, indices)
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

    def simulate(self, strategy, data, strategyParams, fractionOfTotalToTrade=1, step=1):
        account = Account(self.startMoney, self.provision)
        lastValue = 0
        for i in range(self.idxStart, self.idxEnd, step):
            decision = strategy(data, i, strategyParams)
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



def holdStrategy(data, pricesSoFar, params):
    return "BUY"

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

def majorMovingAveragesStrategy(data, i, strategyParams):
    sma20 = data.sma20[i]
    sma50 = data.sma50[i]
    sma100 = data.sma100[i]
    sma200 = data.sma200[i]
    if (data.closes[i] > sma20) and (data.closes[i] > sma50) and (data.closes[i] > sma100) and data.closes[i] > sma200:
        return "BUY"
    elif (data.closes[i] < sma20) and (data.closes[i] < sma50) and (data.closes[i] < sma100) and (data.closes[i] < sma200):
        return "SELL"
    else:
        return "HOLD"

def macdStrategy(data, i, strategyParams):
    macdLine = data.macd[i][0]
    macdDiff = data.macd[i][1]
    macdSignal = data.macd[i][2]
    macdPrevLine = data.macd[i-1][0]
    macdPrevSignal = data.macd[i-1][2]
    if macdLine > 0 and macdPrevLine < 0:
        return "BUY"
    elif macdLine < 0 and macdPrevLine > 0:
        return "SELL"
    else:
        return "HOLD"


def majorMovingAveragesStrategyWeights(data, i, strategyParams):
    sma20 = data.ema20[i]
    sma50 = data.ema50[i]
    sma100 = data.ema100[i]
    sma200 = data.ema200[i]
    weights = strategyParams
    score = 0
    if (data.closes[i] > sma20):
        score += weights[0]
    if (data.closes[i] > sma50):
        score += weights[1]
    if (data.closes[i] > sma100):
        score += weights[2]
    if (data.closes[i] > sma200):
        score += weights[3]
    if (data.closes[i] < sma20):
        score -= weights[0]
    if (data.closes[i] < sma50):
        score -= weights[1]
    if (data.closes[i] < sma100):
        score -= weights[2]
    if (data.closes[i] < sma200):
        score -= weights[3]

    if score > 0.3 * np.sum(weights):
        return "BUY"
    if score < -0.3 * np.sum(weights):
        return "SELL"
    else:
        return "HOLD"

def majorMovingAveragesCrossStrategy(data, i, strategyParams):
    if majorMovingAveragesStrategy(data, i) == "BUY" and majorMovingAveragesStrategy(data, i-1) != "BUY":
        return "BUY"
    elif majorMovingAveragesStrategy(data, i) == "SELL":
        return "SELL"
    else:
        return "HOLD"

def macdAndMovingStrategy(data, i, strategyParams):
    if majorMovingAveragesStrategyWeights(data, i, strategyParams) == "BUY" and macdStrategy(data, i, strategyParams) == "BUY":
        return "BUY"
    elif majorMovingAveragesStrategyWeights(data, i, strategyParams) == "SELL" or macdStrategy(data, i, strategyParams) == "SELL":
        return "SELL"
    else:
        return "HOLD"

def macdAndMovingStrategy2(data, i, strategyParams):
    if majorMovingAveragesStrategyWeights(data, i, strategyParams) == "BUY" or data.macd[i][0] > 0:
        return "BUY"
    elif majorMovingAveragesStrategyWeights(data, i, strategyParams) == "SELL" and data.macd[i][0] < 0:
        return "SELL"
    else:
        return "HOLD"
# majorMovingAveragesStrategyAI(data, i, strategyParams):
#     svm()

class StrategyTester:
    def doSimulation(i, iterations, strategy, data, strategyParams, chunkSize, startDelay):
        [idxStart, idxEnd] = getRandomDataChunk(chunkSize, len(data.closes), startDelay, True)
        # print(f"{i+1}/{iterations} (range {idxStart} - {idxEnd}):")

        simulation = Simulation(data.closes, idxStart, idxEnd - 1, 10000, 0.001, True)
        ratio = simulation.simulate(strategy, data, strategyParams, 1)

        ratioRef = simulation.simulate(holdStrategy, data, strategyParams, 1)
        # print("")
        return [ratio, ratioRef]

    def testStrategy(iterations, strategy, data, strategyParams, chunkSize, startDelay):
        ratios = []
        ratiosRef = []
        for i in range(iterations):
            [ratio, ratioRef] = StrategyTester.doSimulation(i, iterations, strategy, data, strategyParams, chunkSize, startDelay)
            ratios.append(ratio)
            ratiosRef.append(ratioRef)

        averageRatio = sum(ratios)/len(ratios)
        averageRatioRef = sum(ratiosRef)/len(ratiosRef)
        print(f"Average ratio of start money for chosen strategy: {round(averageRatio * 100, 2)}%")
        print(f"Average ratio of start money for holding: {round(sum(ratiosRef)/len(ratiosRef) * 100, 2)}%")
        print(f"Best for chosen strategy: {round((max(ratios)) * 100)}%")
        print(f"Best for holding: {round((max(ratiosRef)) * 100)}%")
        print(f"Worst for chosen strategy: {round((min(ratios)) * 100)}%")
        print(f"Worst for holding: {round((min(ratiosRef)) * 100)}%")
        print("")
        return averageRatio / averageRatioRef



def demonstrate(data, weights):
    chunkSize = daysToIntervals(30)


    startDelay = daysToIntervals(200)

    print("Demonstrating result for chosen paramters...")

    result = StrategyTester.testStrategy(20, majorMovingAveragesStrategyWeights, data, weights, chunkSize, startDelay)
    # result = StrategyTester.testStrategy(100, macdAndMovingStrategy2, data, weights, chunkSize, startDelay)

def train(data):
    chunkSize = daysToIntervals(30)

    startDelay = daysToIntervals(200)

    print("Tuning paramters...")

    solutions = []
    for s in range(100):
        solutions.append((random.randint(0, 5)/5,
                        random.randint(0, 5)/5,
                        random.randint(0, 5)/5,
                        random.randint(0, 5)/5))

    rankedSolutions = []
    i = 0
    for s in solutions:
        result = StrategyTester.testStrategy(20, majorMovingAveragesStrategyWeights, data, s, chunkSize, startDelay)
        rankedSolutions.append( (result, s) )
        rankedSolutions.sort()
        rankedSolutions.reverse()
        i += 1
        print(f"(Test {i}) weights: {s} result: {result}")
        print("")

    bestParams = rankedSolutions[0]

    print(f"Best parameters are: {bestParams}")
    return bestParams

def main():
    # data = readData("./data/hourly/EURUSD60-done.csv", [0, 2, 3, 4, 5])
    data = readData("./data/hourly/eth.csv")

    data.initTechnicals()

    # demonstrate or train

    # train(data)
    demonstrate(data, [1.0, 0.4, 1.0, 0.6])


main()
