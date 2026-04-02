from datetime import date
from pydoc import doc
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
import itertools

# Main function is at the bottom of the file :)

def calc_rsi_wilder(closes, period = 14):
    closes = pd.Series(closes, dtype=float).reset_index(drop = True)
    rsi = pd.Series(np.nan, index = closes.index, dtype=float)

    if len(closes) <= period:
        return rsi

    price_diff = closes.diff()
    gains = price_diff.clip(lower = 0).fillna(0.0)
    losses = (-price_diff.clip(upper = 0)).fillna(0.0)

    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()

    if avg_loss == 0:
        rsi.iloc[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        relative_strength = avg_gain / avg_loss
        rsi.iloc[period] = 100 - (100 / (1 + relative_strength))

    for i in range(period + 1, len(closes)):
        avg_gain = ((avg_gain * (period - 1)) + gains.iloc[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses.iloc[i]) / period

        if avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
            continue

        relative_strength = avg_gain / avg_loss
        rsi.iloc[i] = 100 - (100 / (1 + relative_strength))

    return rsi

# unix, date, symbol, open, high, low, close, VolumeBTC, VolumeUSDT, tradecount
class DataPoints:
    def __init__(self, rawData, indices):
        self.dates = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
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
        for row in reversed(rawData):
            if row[0] == '#':
                continue
            row = row.split(",")
            self.dates.append(row[indices[0]])
            self.opens.append(float(row[indices[1]]))
            self.highs.append(float(row[indices[2]]))
            self.lows.append(float(row[indices[3]]))
            self.closes.append(float(row[indices[4]]))

    # def printDataPoint(self):
        # print(f"Date: {self.date} open: {self.open} h: {self.high} l: {self.low} close: {self.close}")
    
    def initTechnicals(self):
        print(f"Calculating technicals...")
        closes = pd.Series(self.closes, dtype=float)

        self.sma20 = closes.rolling(window = 20, min_periods = 20).mean().to_numpy()
        self.sma50 = closes.rolling(window = 50, min_periods = 50).mean().to_numpy()
        self.sma100 = closes.rolling(window = 100, min_periods = 100).mean().to_numpy()
        self.sma200 = closes.rolling(window = 200, min_periods = 200).mean().to_numpy()

        self.ema20 = closes.ewm(span = 20, adjust = False, min_periods = 20).mean().to_numpy()
        self.ema50 = closes.ewm(span = 50, adjust = False, min_periods = 50).mean().to_numpy()
        self.ema100 = closes.ewm(span = 100, adjust = False, min_periods = 100).mean().to_numpy()
        self.ema200 = closes.ewm(span = 200, adjust = False, min_periods = 200).mean().to_numpy()

        self.rsi = calc_rsi_wilder(closes, 14).to_numpy()

        macd_fast = closes.ewm(span = 14, adjust = False, min_periods = 14).mean()
        macd_slow = closes.ewm(span = 28, adjust = False, min_periods = 28).mean()
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm(span = 9, adjust = False, min_periods = 9).mean()
        macd_histogram = macd_line - macd_signal
        self.macd = np.column_stack((
            macd_line.to_numpy(),
            macd_histogram.to_numpy(),
            macd_signal.to_numpy(),
        ))

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

    def displayResultMsg(self, msg, start, end, lastValue, ratio, trades):
        percentage = round(ratio * 100)
        if len(msg) == 0:
            msg = "Result"
        print(f"{msg} ({start} - {end}): Na koniec po {trades} tranzakcjach masz: {round(lastValue)} dol z zainwestowanych {self.startMoney}"
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
            self.displayResultMsg(strategy.__name__, self.idxStart, self.idxEnd, lastValue, ratio, self.trades)
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


def weightedMajorEmasStrategy(data, i, strategyParams):
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

def majorMovingAveragesCrossStrategy(data, i, strategyParams):
    if majorMovingAveragesStrategy(data, i) == "BUY" and majorMovingAveragesStrategy(data, i-1) != "BUY":
        return "BUY"
    elif majorMovingAveragesStrategy(data, i) == "SELL":
        return "SELL"
    else:
        return "HOLD"

def macdAndMovingStrategy(data, i, strategyParams):
    if weightedMajorEmasStrategy(data, i, strategyParams) == "BUY" and macdStrategy(data, i, strategyParams) == "BUY":
        return "BUY"
    elif weightedMajorEmasStrategy(data, i, strategyParams) == "SELL" or macdStrategy(data, i, strategyParams) == "SELL":
        return "SELL"
    else:
        return "HOLD"

def macdAndMovingStrategy2(data, i, strategyParams):
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

def emaOrderStrategy(data, i, strategyParams):
    priceAvgs = [(data.closes[i], 1), (data.ema20[i], 2), (data.ema50[i], 3), (data.ema100[i], 4), (data.ema200[i], 5)]
    priceAvgs.sort(reverse = True)
    ranks = []
    for priceAvg in priceAvgs:
        ranks.append(priceAvg[1])
    return strategyParams[tuple(ranks)]


class StrategyTester:
    def doSimulation(i, iterations, strategy, data, provision, strategyParams, chunkSize, startDelay, isSilent):
        [idxStart, idxEnd] = getRandomDataChunk(chunkSize, len(data.closes), startDelay, True)
        # print(f"{i+1}/{iterations} (range {idxStart} - {idxEnd}):")

        simulation = Simulation(data.closes, idxStart, idxEnd - 1, 10000, provision, isSilent)
        ratio = simulation.simulate(strategy, data, strategyParams, 1)

        ratioRef = simulation.simulate(holdStrategy, data, strategyParams, 1)
        # print("")
        return [ratio, ratioRef]

    def testStrategy(iterations, strategy, data, provision, strategyParams, chunkSize, startDelay, isSilent):
        ratios = []
        ratiosRef = []
        for i in range(iterations):
            [ratio, ratioRef] = StrategyTester.doSimulation(
                    i, iterations, strategy, data, provision, strategyParams, chunkSize, startDelay, isSilent
            )
            ratios.append(ratio)
            ratiosRef.append(ratioRef)

        averageRatio = sum(ratios)/len(ratios)
        averageRatioRef = sum(ratiosRef)/len(ratiosRef)
        print("")
        print(f"Average ratio of start money for chosen strategy: {round(averageRatio * 100, 2)}%")
        print(f"Average ratio of start money for holding: {round(sum(ratiosRef)/len(ratiosRef) * 100, 2)}%")
        print(f"Best for chosen strategy: {round((max(ratios)) * 100)}%")
        print(f"Best for holding: {round((max(ratiosRef)) * 100)}%")
        print(f"Worst for chosen strategy: {round((min(ratios)) * 100)}%")
        print(f"Worst for holding: {round((min(ratiosRef)) * 100)}%")
        print("")
        return averageRatio / averageRatioRef


def demonstrate(data, provision, strategy, params):
    chunkSize = daysToIntervals(300)
    startDelay = 200
    print("Demonstrating result for chosen parameters...")
    result = StrategyTester.testStrategy(100, strategy, data, provision,params, chunkSize, startDelay, False)

def train(data, provision, strategy):
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
        result = StrategyTester.testStrategy(20, strategy, data, provision, s, chunkSize, startDelay, True)
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
    data = readData("./data/hourly/EURUSD60-done.csv", [0, 2, 3, 4, 5])
    # data = readData("./data/hourly/eth.csv")
    data.initTechnicals()

    plt.plot(data.closes)
    # plt.show()

    training = False

    provision = 0.001
    

    if (training):
        train(data, provision, majorMovingAveragesStrategy)
    else:
        demonstrate(data, provision, majorMovingAveragesStrategy, [])

main()
