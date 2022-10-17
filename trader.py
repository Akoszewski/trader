from datetime import date
from pydoc import doc
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
import pandas_ta as pta
import itertools
import numba as nb
from typing import List
from numba.experimental import jitclass

# Main function is at the bottom of the file :)

def getSlice(N, splitRatio, isTest):
    if isTest:
        begin = int(N * splitRatio)
        end = N
    else:
        begin = 0
        end = int(N * splitRatio)
    return [i for i in range(end-1, begin-1, -1)]

# unix, date, symbol, open, high, low, close, VolumeBTC, VolumeUSDT, tradecount
spec = [
    ('opens', nb.float64[:]),
    ('highs', nb.float64[:]),
    ('lows', nb.float64[:]),
    ('closes', nb.float64[:]),
    ('sma20', nb.float64[:]),
    ('sma50', nb.float64[:]),
    ('sma100', nb.float64[:]),
    ('sma200', nb.float64[:]),
    ('ema20', nb.float64[:]),
    ('ema50', nb.float64[:]),
    ('ema100', nb.float64[:]),
    ('ema200', nb.float64[:]),
    ('rsi', nb.float64[:]),
    ('macd', nb.float64[:]),
]
@jitclass(spec)
class DataPointsJit:
    def __init__(self):
        self.opens = np.empty(0, dtype=np.float64)
        self.highs = np.empty(0, dtype=np.float64)
        self.lows = np.empty(0, dtype=np.float64)
        self.closes = np.empty(0, dtype=np.float64)
        self.sma20 = np.empty(0, dtype=np.float64)
        self.sma50 = np.empty(0, dtype=np.float64)
        self.sma100 = np.empty(0, dtype=np.float64)
        self.sma200 = np.empty(0, dtype=np.float64)
        self.ema20 = np.empty(0, dtype=np.float64)
        self.ema50 = np.empty(0, dtype=np.float64)
        self.ema100 = np.empty(0, dtype=np.float64)
        self.ema200 = np.empty(0, dtype=np.float64)
        self.rsi = np.empty(0, dtype=nb.float64)
        self.macd = np.empty(0, dtype=np.float64)

def createDataPoints(rawData : List[str], indices, splitRatio, isTest) -> DataPointsJit:
    opens = []
    highs = []
    lows = []
    closes = []
    for i in getSlice(len(rawData), splitRatio, isTest):
        row = rawData[i]
        if row[0] == '#':
            continue
        row = row.split(",")
        opens.append(float(row[indices[1]]))
        highs.append(float(row[indices[2]]))
        lows.append(float(row[indices[3]]))
        closes.append(float(row[indices[4]]))

    dataPointsJit = DataPointsJit()

    df = pd.DataFrame(closes, columns = ['closes'])
    dataPointsJit.sma20 = pta.sma(close = df['closes'], length = 20).to_numpy()
    dataPointsJit.sma50 = pta.sma(close = df['closes'], length = 50).to_numpy()
    dataPointsJit.sma100 = pta.sma(close = df['closes'], length = 100).to_numpy()
    dataPointsJit.sma200 = pta.sma(close = df['closes'], length = 200).to_numpy()
    dataPointsJit.ema20 = pta.ema(close = df['closes'], length = 20).to_numpy()
    dataPointsJit.ema50 = pta.ema(close = df['closes'], length = 50).to_numpy()
    dataPointsJit.ema100 = pta.ema(close = df['closes'], length = 100).to_numpy()
    dataPointsJit.ema200 = pta.ema(close = df['closes'], length = 200).to_numpy()
    dataPointsJit.rsi = pta.rsi(close = df['closes'], length = 14).to_numpy()
    # dataPointsJit.macd = pta.macd(close = df['closes'], length = 14).to_numpy()

    dataPointsJit.opens = np.array(opens, dtype=np.float64)
    dataPointsJit.highs = np.array(highs, dtype=np.float64)
    dataPointsJit.lows = np.array(lows, dtype=np.float64)
    dataPointsJit.closes = np.array(closes, dtype=np.float64)

    return dataPointsJit

def readData(filename, indices = [1,3,4,5,6], splitRatio = 1, isTest = False) -> DataPointsJit:
    data = []
    with open(filename, "r") as f:
        rawData = f.readlines()
        data = createDataPoints(rawData, indices, splitRatio, isTest)
    return data

spec = [
    ('startMoney', nb.float64),
    ('money', nb.float64),
    ('stock', nb.float64),
    ('provision', nb.float64),
    ('trades', nb.int32),
]
@jitclass(spec)
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
    def __init__(self, prices, idxStart, idxEnd, startMoney, provision = 0, silenceLevel = 1):
        self.prices = prices
        self.startMoney = startMoney
        self.provision = provision
        self.idxStart = idxStart
        self.idxEnd = idxEnd
        self.silenceLevel = silenceLevel
        self.trades = 0

    def displayResultMsg(self, msg, start, end, lastValue, ratio, trades):
        percentage = round(ratio * 100)
        if len(msg) == 0:
            msg = "Result"
        print(f"{msg} ({start} - {end}): Na koniec po {trades} tranzakcjach masz: {round(lastValue)} dol z zainwestowanych {self.startMoney}"
                f" czyli {percentage}%")

    def simulate(self, strategy, data : DataPointsJit, strategyParams, fractionOfTotalToTrade=1, step=1):
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
        if self.silenceLevel == 0:
            self.displayResultMsg(strategy.__name__, self.idxStart, self.idxEnd, lastValue, ratio, self.trades)
        return ratio



def holdStrategy(data : DataPointsJit, pricesSoFar, params):
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
    ema20 = data.ema20[i]
    ema50 = data.ema50[i]
    ema100 = data.ema100[i]
    ema200 = data.ema200[i]
    if (data.closes[i] > ema20) and (data.closes[i] > ema50) and (data.closes[i] > ema100) and data.closes[i] > ema200:
        return "BUY"
    elif (data.closes[i] < ema20) and (data.closes[i] < ema50) and (data.closes[i] < ema100) and (data.closes[i] < ema200):
        return "SELL"
    else:
        return "HOLD"

def movingAveragesStrategy(data, i, strategyParams):
    ema1 = data.ema20[i]
    ema2 = data.ema100[i]
    if (data.closes[i] > ema1) and (data.closes[i] > ema2) and (ema1 > ema2):
        return "BUY"
    elif (data.closes[i] < ema1) and (data.closes[i] < ema2) and (ema1 < ema2):
        return "SELL"
    else:
        return "HOLD"

def movingAveragesStrategyIncreasing(data, i, strategyParams):
    ema1 = data.ema20[i]
    ema2 = data.ema100[i]
    prevEma2 = data.ema100[i-5]
    if (data.closes[i] > ema1) and (data.closes[i] > ema2) and (ema1 > ema2) and prevEma2 < ema2:
        return "BUY"
    elif (data.closes[i] < ema1) and (data.closes[i] < ema2) and (ema1 < ema2):
        return "SELL"
    else:
        return "HOLD"

def macdStrategy(data : DataPointsJit, i, strategyParams):
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


def weightedMajorEmasStrategy(data : DataPointsJit, i, strategyParams):
    ema20 = data.ema20[i]
    ema50 = data.ema50[i]
    ema100 = data.ema100[i]
    ema200 = data.ema200[i]
    weights = strategyParams[2:]
    score = 0
    if (data.closes[i] > ema20):
        score += weights[0]
    if (data.closes[i] > ema50):
        score += weights[1]
    if (data.closes[i] > ema100):
        score += weights[2]
    if (data.closes[i] > ema200):
        score += weights[3]
    if (data.closes[i] < ema20):
        score -= weights[0]
    if (data.closes[i] < ema50):
        score -= weights[1]
    if (data.closes[i] < ema100):
        score -= weights[2]
    if (data.closes[i] < ema200):
        score -= weights[3]

    if score > strategyParams[0] * np.sum(weights):
        return "BUY"
    if score < strategyParams[1] * np.sum(weights):
        return "SELL"
    else:
        return "HOLD"

def majorMovingAveragesCrossStrategy(data : DataPointsJit, i, strategyParams):
    if majorMovingAveragesStrategy(data, i) == "BUY" and majorMovingAveragesStrategy(data, i-1) != "BUY":
        return "BUY"
    elif majorMovingAveragesStrategy(data, i) == "SELL":
        return "SELL"
    else:
        return "HOLD"

def macdAndMovingStrategy(data : DataPointsJit, i, strategyParams):
    if weightedMajorEmasStrategy(data, i, strategyParams) == "BUY" and macdStrategy(data, i, strategyParams) == "BUY":
        return "BUY"
    elif weightedMajorEmasStrategy(data, i, strategyParams) == "SELL" or macdStrategy(data, i, strategyParams) == "SELL":
        return "SELL"
    else:
        return "HOLD"

def macdAndMovingStrategy2(data : DataPointsJit, i, strategyParams):
    if weightedMajorEmasStrategy(data, i, strategyParams) == "BUY" or data.macd[i][0] > 0:
        return "BUY"
    elif weightedMajorEmasStrategy(data, i, strategyParams) == "SELL" and data.macd[i][0] < 0:
        return "SELL"
    else:
        return "HOLD"

# majorMovingAveragesStrategyAI(data, i, strategyParams):
#     svm()

def genRandomEmaOrderStrategyParams():
    possibleRanks = [1, 2, 3, 4, 5]
    rankPermutations = itertools.permutations(possibleRanks)
    params = dict()
    for permutation in rankPermutations:
        params[permutation] = random.choice(["BUY", "SELL", "HOLD"])
    return params

def emaOrderStrategy(data : DataPointsJit, i, strategyParams):
    priceAvgs = [(data.closes[i], 1), (data.ema20[i], 2), (data.ema50[i], 3), (data.ema100[i], 4), (data.ema200[i], 5)]
    priceAvgs.sort(reverse = True)
    ranks = []
    for priceAvg in priceAvgs:
        ranks.append(priceAvg[1])
    return strategyParams[tuple(ranks)]


class StrategyTester:
    def doSimulation(i, iterations, strategy, data : DataPointsJit, strategyParams, chunkSize, startDelay, silenceLevel):
        [idxStart, idxEnd] = getRandomDataChunk(chunkSize, len(data.closes), startDelay, True)
        # print(f"{i+1}/{iterations} (range {idxStart} - {idxEnd}):")

        simulation = Simulation(data.closes, idxStart, idxEnd - 1, 10000, 0.001, silenceLevel)
        ratio = simulation.simulate(strategy, data, strategyParams, 1)

        ratioRef = simulation.simulate(holdStrategy, data, strategyParams, 1)
        # print("")
        return [ratio, ratioRef]

    def testStrategy(iterations, strategy, data : DataPointsJit, strategyParams, chunkSize, startDelay, silenceLevel):
        ratios = []
        ratiosRef = []
        for i in range(iterations):
            [ratio, ratioRef] = StrategyTester.doSimulation(
                    i, iterations, strategy, data, strategyParams, chunkSize, startDelay, silenceLevel
            )
            ratios.append(ratio)
            ratiosRef.append(ratioRef)

        averageRatio = sum(ratios)/len(ratios)
        averageRatioRef = sum(ratiosRef)/len(ratiosRef)
        if silenceLevel <= 1:
            print("")
            print(f"Average ratio of start money for chosen strategy: {round(averageRatio * 100, 2)}%")
            print(f"Average ratio of start money for holding: {round(sum(ratiosRef)/len(ratiosRef) * 100, 2)}%")
            print(f"Best for chosen strategy: {round((max(ratios)) * 100)}%")
            print(f"Best for holding: {round((max(ratiosRef)) * 100)}%")
            print(f"Worst for chosen strategy: {round((min(ratios)) * 100)}%")
            print(f"Worst for holding: {round((min(ratiosRef)) * 100)}%")
            print("")
        return averageRatio / averageRatioRef



def demonstrate(data : DataPointsJit, strategy, params, silenceLevel):
    chunkSize = daysToIntervals(300)
    startDelay = 200
    print("Demonstrating result for chosen parameters...")
    result = StrategyTester.testStrategy(20, strategy, data, params, chunkSize, startDelay, silenceLevel)

def train(data : DataPointsJit, strategy):
    chunkSize = daysToIntervals(300)
    startDelay = 201

    print("Tuning parameters...")

    solutions = []
    for s in range(100):
        solutions.append((
                        0.3,
                       -0.3,
                        random.randint(0, 5)/5,
                        random.randint(0, 5)/5,
                        random.randint(0, 5)/5,
                        random.randint(0, 5)/5))

    rankedSolutions = []
    i = 0
    for s in solutions:
        result = StrategyTester.testStrategy(20, strategy, data, s, chunkSize, startDelay, True)
        rankedSolutions.append( (result, s) )
        rankedSolutions.sort()
        rankedSolutions.reverse()
        i += 1
        print(f"(Test {i}) parameters: {s} result: {result}")
        print("")

    bestParams = rankedSolutions[0]

    print(f"Best parameters are: {bestParams}")
    return bestParams

def main():
    # data = readData("./data/hourly/EURUSD60-done.csv", [0, 2, 3, 4, 5])
    data = readData("./data/hourly/eth.csv")

    plt.plot(data.closes)
    # plt.show()

    training = False
    silenceLevel = 1 # 0->verbose, 2->mute

    if (training):
        train(data, weightedMajorEmasStrategy)
    else:
        params = [0.3, -0.3, 1.0, 0.4, 1.0, 0.6]
        params2 = (0.3, -0.3, 0.0, 0.8, 0.0, 1.0)
        paramsSafest = [0.3, -0.3, 0.45134, 0.6169, 0.39278, 0.788256]
        paramsCmaes = [0.6384061662936363, -0.6619728111749704, 1.124445707982012, 0.15998907010113295, 2.3330288361768816, -0.6894718752979803]         
        demonstrate(data, weightedMajorEmasStrategy, paramsSafest, silenceLevel)

if __name__ == '__main__':
    main()
