from datetime import date
from pydoc import doc
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
import itertools

# Main function is at the bottom of the file :)

BASE_EMA_PERIODS = (20, 50, 100, 200)
MAX_EMA_MULTIPLIER = 3.0

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


def getRowValue(row, indexSpec):
    if isinstance(indexSpec, (tuple, list)):
        return " ".join(row[index].strip() for index in indexSpec)
    return row[indexSpec].strip()

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
        self.emaCache = dict()
        self.rsi = []
        self.macd = []
        parsedRows = []
        for rawRow in rawData:
            if len(rawRow) == 0 or rawRow[0] == '#':
                continue
            parsedRows.append(rawRow.strip().split(","))

        if len(parsedRows) >= 2:
            firstTimestamp = getRowValue(parsedRows[0], indices[0])
            lastTimestamp = getRowValue(parsedRows[-1], indices[0])
            if firstTimestamp > lastTimestamp:
                parsedRows.reverse()

        for row in parsedRows:
            self.dates.append(getRowValue(row, indices[0]))
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
        self.emaCache = {
            20: self.ema20,
            50: self.ema50,
            100: self.ema100,
            200: self.ema200,
        }

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

    def getEma(self, period):
        if period not in self.emaCache:
            closes = pd.Series(self.closes, dtype=float)
            self.emaCache[period] = closes.ewm(span = period, adjust = False, min_periods = period).mean().to_numpy()
        return self.emaCache[period]

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

def getRandomDataChunkInRange(minChunkLength, rangeStart, rangeEnd, isChunkLengthFixed = False):
    diff = minChunkLength
    if rangeEnd - rangeStart < diff:
        raise ValueError(f"Range [{rangeStart}, {rangeEnd}) is too small for chunk length {diff}")

    left = random.randint(rangeStart, rangeEnd - diff)
    if isChunkLengthFixed:
        right = left + diff
    else:
        right = random.randint(left + diff, rangeEnd)
    return [left, right]


def daysToIntervals(days, intervalsPerDay = 24):
    return math.floor(days * intervalsPerDay)

def isInvalidTechnicalValue(value):
    return pd.isna(value)


def getWeightedEmaMultiplier(strategyParams):
    if len(strategyParams) >= 7:
        return strategyParams[6]
    return 1.0


def getWeightedEmaPeriods(strategyParams):
    multiplier = getWeightedEmaMultiplier(strategyParams)
    scaledPeriods = []
    previousPeriod = 0
    for basePeriod in BASE_EMA_PERIODS:
        scaledPeriod = max(2, int(round(basePeriod * multiplier)))
        scaledPeriod = max(scaledPeriod, previousPeriod + 1)
        scaledPeriods.append(scaledPeriod)
        previousPeriod = scaledPeriod
    return tuple(scaledPeriods)


def getWeightedEmaWarmup(strategyParams):
    return getWeightedEmaPeriods(strategyParams)[-1]

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

def movingAveragesCrossStrategy(data, i, strategyParams):
    if i <= 0:
        return "HOLD"

    fastEma = data.ema20[i]
    slowEma = data.ema100[i]
    prevFastEma = data.ema20[i - 1]
    prevSlowEma = data.ema100[i - 1]

    if isInvalidTechnicalValue(fastEma) or isInvalidTechnicalValue(slowEma):
        return "HOLD"
    if isInvalidTechnicalValue(prevFastEma) or isInvalidTechnicalValue(prevSlowEma):
        return "HOLD"

    if prevFastEma <= prevSlowEma and fastEma > slowEma:
        return "BUY"
    elif prevFastEma >= prevSlowEma and fastEma < slowEma:
        return "SELL"
    else:
        return "HOLD"

def movingAveragesStrategyIncreasing(data, i, strategyParams):
    if i < 5:
        return "HOLD"

    ema1 = data.ema20[i]
    ema2 = data.ema100[i]
    prevEma2 = data.ema100[i-5]

    if isInvalidTechnicalValue(ema1) or isInvalidTechnicalValue(ema2) or isInvalidTechnicalValue(prevEma2):
        return "HOLD"

    if (data.closes[i] > ema1) and (data.closes[i] > ema2) and (ema1 > ema2) and prevEma2 < ema2:
        return "BUY"
    elif (data.closes[i] < ema1) and (data.closes[i] < ema2) and (ema1 < ema2):
        return "SELL"
    else:
        return "HOLD"

def macdStrategy(data, i, strategyParams):
    if i <= 0:
        return "HOLD"

    macdLine = data.macd[i][0]
    macdDiff = data.macd[i][1]
    macdSignal = data.macd[i][2]
    macdPrevLine = data.macd[i-1][0]
    macdPrevSignal = data.macd[i-1][2]

    if isInvalidTechnicalValue(macdLine) or isInvalidTechnicalValue(macdPrevLine):
        return "HOLD"

    if macdLine > 0 and macdPrevLine < 0:
        return "BUY"
    elif macdLine < 0 and macdPrevLine > 0:
        return "SELL"
    else:
        return "HOLD"

def rsiStrategy(data, i, strategyParams):
    buyThreshold = 30
    sellThreshold = 70

    if len(strategyParams) >= 2:
        buyThreshold = strategyParams[0]
        sellThreshold = strategyParams[1]

    if i <= 0:
        return "HOLD"

    rsi = data.rsi[i]
    prevRsi = data.rsi[i - 1]

    if isInvalidTechnicalValue(rsi) or isInvalidTechnicalValue(prevRsi):
        return "HOLD"

    if prevRsi <= buyThreshold and rsi > buyThreshold:
        return "BUY"
    elif prevRsi >= sellThreshold and rsi < sellThreshold:
        return "SELL"
    else:
        return "HOLD"


def weightedMajorEmasStrategy(data, i, strategyParams):
    emaPeriods = getWeightedEmaPeriods(strategyParams)
    ema20 = data.getEma(emaPeriods[0])[i]
    ema50 = data.getEma(emaPeriods[1])[i]
    ema100 = data.getEma(emaPeriods[2])[i]
    ema200 = data.getEma(emaPeriods[3])[i]

    if isInvalidTechnicalValue(ema20) or isInvalidTechnicalValue(ema50):
        return "HOLD"
    if isInvalidTechnicalValue(ema100) or isInvalidTechnicalValue(ema200):
        return "HOLD"

    weights = strategyParams[2:6]
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
    def doSimulation(i, iterations, strategy, data, provision, strategyParams, chunkSize, startDelay, isSilent, rangeStart = None, rangeEnd = None):
        if rangeStart is None:
            rangeStart = startDelay
        else:
            rangeStart = max(rangeStart, startDelay)

        if rangeEnd is None:
            rangeEnd = len(data.closes)

        [idxStart, idxEnd] = getRandomDataChunkInRange(chunkSize, rangeStart, rangeEnd, True)
        # print(f"{i+1}/{iterations} (range {idxStart} - {idxEnd}):")

        simulation = Simulation(data.closes, idxStart, idxEnd - 1, 10000, provision, isSilent)
        ratio = simulation.simulate(strategy, data, strategyParams, 1)

        ratioRef = simulation.simulate(holdStrategy, data, strategyParams, 1)
        # print("")
        return [ratio, ratioRef]

    def testStrategy(iterations, strategy, data, provision, strategyParams, chunkSize, startDelay, isSilent, rangeStart = None, rangeEnd = None):
        ratios = []
        ratiosRef = []
        for i in range(iterations):
            [ratio, ratioRef] = StrategyTester.doSimulation(
                    i, iterations, strategy, data, provision, strategyParams, chunkSize, startDelay, isSilent, rangeStart, rangeEnd
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

    def getTrainTestRanges(dataLength, startDelay, splitRatio = 0.5):
        usableStart = startDelay
        usableEnd = dataLength
        usableLength = usableEnd - usableStart
        splitPoint = usableStart + math.floor(usableLength * splitRatio)
        return (usableStart, splitPoint), (splitPoint, usableEnd)


def printTrainingResult(label, trainingResult):
    trainScore, testScore, params = trainingResult
    print(label)
    print(f"  Train score: {round(trainScore, 4)}")
    print(f"  Test score:  {round(testScore, 4)}")
    print(f"  Params:      {params}")
    print(f"  EMA periods: {getWeightedEmaPeriods(params)}")
    print("")


def scoreTrainingResult(trainingResult):
    trainScore, testScore, _ = trainingResult
    return min(trainScore, testScore) - 0.1 * abs(trainScore - testScore)


def genRandomWeightedEmaParams():
    return (
        round(random.uniform(0.2, 1.0), 3),
        round(random.uniform(-1.0, -0.2), 3),
        round(random.uniform(0.0, 1.0), 3),
        round(random.uniform(0.0, 1.0), 3),
        round(random.uniform(0.0, 1.0), 3),
        round(random.uniform(0.0, 1.0), 3),
        round(random.uniform(0.5, MAX_EMA_MULTIPLIER), 3),
    )


def clampWeightedEmaParams(params):
    buyThreshold = min(max(params[0], 0.0), 1.0)
    sellThreshold = min(max(params[1], -1.0), 0.0)
    weights = [min(max(weight, 0.0), 1.0) for weight in params[2:6]]
    emaMultiplier = min(max(params[6], 0.5), MAX_EMA_MULTIPLIER)
    return (
        round(buyThreshold, 3),
        round(sellThreshold, 3),
        round(weights[0], 3),
        round(weights[1], 3),
        round(weights[2], 3),
        round(weights[3], 3),
        round(emaMultiplier, 3),
    )


def mutateWeightedEmaParams(params, stepScale):
    mutated = [
        params[0] + random.uniform(-stepScale, stepScale),
        params[1] + random.uniform(-stepScale, stepScale),
    ]

    for weight in params[2:]:
        mutated.append(weight + random.uniform(-stepScale, stepScale))

    return clampWeightedEmaParams(tuple(mutated))


def evaluateWeightedEmaParams(data, provision, params, chunkSize, startDelay, trainStart, trainEnd, testStart, testEnd, isSilent = True):
    trainResult = StrategyTester.testStrategy(20, weightedMajorEmasStrategy, data, provision, params, chunkSize, startDelay, isSilent, trainStart, trainEnd)
    testResult = StrategyTester.testStrategy(20, weightedMajorEmasStrategy, data, provision, params, chunkSize, startDelay, isSilent, testStart, testEnd)
    return (trainResult, testResult, params)


def demonstrate(data, provision, strategy, params):
    chunkSize = daysToIntervals(300)
    startDelay = max(200, getWeightedEmaWarmup(params))
    print("Demonstrating result for chosen parameters...")
    result = StrategyTester.testStrategy(100, strategy, data, provision, params, chunkSize, startDelay, False)

def train(data, provision, strategy):
    chunkSize = daysToIntervals(300)
    startDelay = max(201, math.ceil(BASE_EMA_PERIODS[-1] * MAX_EMA_MULTIPLIER))
    (trainStart, trainEnd), (testStart, testEnd) = StrategyTester.getTrainTestRanges(len(data.closes), startDelay)

    print("Tuning parameters...")
    print(f"Training range: {trainStart} - {trainEnd}")
    print(f"Test range: {testStart} - {testEnd}")

    if strategy != weightedMajorEmasStrategy:
        raise ValueError("train() currently supports weightedMajorEmasStrategy only")

    seedCount = 24
    localRestarts = 5
    localSteps = 18

    rankedSolutions = []
    for seedIndex in range(seedCount):
        params = genRandomWeightedEmaParams()
        result = evaluateWeightedEmaParams(data, provision, params, chunkSize, startDelay, trainStart, trainEnd, testStart, testEnd, True)
        rankedSolutions.append(result)
        print(f"(Seed {seedIndex + 1}) parameters: {params} objective: {round(scoreTrainingResult(result), 4)}")

    rankedSolutions.sort(key = scoreTrainingResult, reverse = True)
    bestSeeds = rankedSolutions[:localRestarts]

    improvedSolutions = list(rankedSolutions)
    for restartIndex, seedResult in enumerate(bestSeeds):
        currentResult = seedResult
        stepScale = 0.25
        print("")
        print(f"Local search {restartIndex + 1} starting from {currentResult[2]}")
        for stepIndex in range(localSteps):
            candidateParams = mutateWeightedEmaParams(currentResult[2], stepScale)
            candidateResult = evaluateWeightedEmaParams(data, provision, candidateParams, chunkSize, startDelay, trainStart, trainEnd, testStart, testEnd, True)
            if scoreTrainingResult(candidateResult) > scoreTrainingResult(currentResult):
                currentResult = candidateResult
                print(f"  Step {stepIndex + 1}: improved to objective {round(scoreTrainingResult(currentResult), 4)} params {currentResult[2]}")
            stepScale *= 0.9
        improvedSolutions.append(currentResult)

    improvedSolutions.sort(key = scoreTrainingResult, reverse = True)
    print("")
    print("Top results:")
    for rank, result in enumerate(improvedSolutions[:5], start = 1):
        trainResult, testResult, params = result
        print(f"  {rank}. objective={round(scoreTrainingResult(result), 4)} train={round(trainResult, 4)} test={round(testResult, 4)} params={params}")

    bestResult = improvedSolutions[0]
    printTrainingResult("Best result:", bestResult)
    return bestResult

def main():
    # data = readData("./data/hourly/EURUSD60-done.csv", [(0, 1), 2, 3, 4, 5])
    data = readData("./data/hourly/btc.csv")
    data.initTechnicals()

    plt.plot(data.closes)
    # plt.show()

    training = True

    provision = 0.002

    trainedResultBestForCryptoZeroProv = (1.314876519201772, 1.4190971357643607, (0.3, -0.3, 0.0, 0.8, 0.8, 0.2))
    trainScore, testScore, paramsForCryptoZeroProv = trainedResultBestForCryptoZeroProv
    
    paramsForCryptoIncreasedTreshold = (0.3, -0.3, 0.0, 0.8, 0.8, 0.2)
    paramsForCryptoWithProv = (0.505, -1.0, 0.294, 0.555, 0.466, 0.365)

    if (training):
        bestResult = train(data, provision, weightedMajorEmasStrategy)
        printTrainingResult("Selected result:", bestResult)
    else:
        demonstrate(data, provision, weightedMajorEmasStrategy, paramsForCryptoWithProv)

if __name__ == "__main__":
    main()
