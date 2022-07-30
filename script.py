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
        self.crypto = 0
        self.provision = 0.001

    def buy(self, amountOfMoney, price):
        if self.money >= amountOfMoney:
            self.money -= amountOfMoney
            self.crypto += (1-self.provision)*amountOfMoney/price
    
    def sell(self, amountOfMoney, price):
        if self.crypto >= amountOfMoney/price:
            self.money += (1-self.provision)*amountOfMoney
            self.crypto -= amountOfMoney/price

    def totalValue(self, price):
        return self.money + self.crypto * price


def allInStrategy(startBalance, x):
    return "BUY"

# Codzienie kupuje za x dolcow
def randomBuySellStrategy(startBalance, x):
    return random.randint(0, 1)

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
    if (rsi > 70):
        return "SELL"
    elif (rsi < 30):
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

def simulateRegularlyEqualBuySell(startBalance, x, strategy):
    totalValues = []
    simulation = Simulation(startBalance)
    prices = [o.close for o in data]
    for i in range(15, len(prices)):
        decision = strategy(prices[:i], 15)
        if (decision == "BUY"):
            simulation.buy(x, data[i].open)
        elif (decision == "SELL"):
            simulation.sell(x, data[i].open)
        totalValues.append(simulation.totalValue(data[i].open))
    return totalValues

def predictNext(prices):
    correct = 0
    wrong = 0
    for i in range(15, len(prices) - 1):
        decision = rsiStrategy(prices[:i], 15)
        if prices[i+1] > prices[i] and decision:
            correct = correct + 1
        elif prices[i+1] > prices[i] and not decision:
            wrong = wrong + 1
        elif prices[i+1] < prices[i] and not decision:
           correct = correct + 1
        else:
            wrong = wrong + 1
    return [correct, wrong]

def simulateRegularDecision(startBalance, strategy, step):
    totalValues = []
    cryptoValues = []
    moneyValues = []
    simulation = Simulation(startBalance)
    prices = [o.close for o in data]
    for i in range(15, len(prices), step):
        decision = strategy(prices[:i], 15)
        decision2 = movingAveragesStrategy(prices[:i], 15)
        if (decision == "BUY" or decision2 == "BUY"):
            simulation.buy(simulation.money, data[i].close)
            print(f"Buy. Curr crypto: {simulation.crypto} curr money: {simulation.money}")
        elif (decision == "SELL" and decision2 == "SELL"):
            simulation.sell(simulation.crypto*data[i].close, data[i].close)
            print(f"Sell. Curr crypto: {simulation.crypto} curr money: {simulation.money}")
        totalValues.append(simulation.totalValue(data[i].close))
        cryptoValues.append(simulation.crypto * data[i].close)
        moneyValues.append(simulation.money)
    print(f"Crypto: {simulation.crypto} money: {simulation.money}")
    return [totalValues, moneyValues, cryptoValues]


# data = readData("btc_every_h.csv")
data = readData("HistoricalPrices.csv", [0,1,2,3,4])
data = data[::]
# data = data[int(len(data)*0.7):]
plt.figure()
plotData(data)
startBalance = 1000000
[totalValues, moneyValues, cryptoValues] = simulateRegularDecision(startBalance, rsiStrategy, 1)
print(f"Na koniec masz: {round(totalValues[-1])} dol z zainwestowanych {startBalance}")

# plt.plot(totalValues, moneyValues, cryptoValues)
plt.figure()
plt.plot(range(len(data) - 15), totalValues, cryptoValues)

plt.show()

# prices = [o.close for o in data]
# [corr, wrong] = simulateRegularDecision(prices)
# print(f"correct: {corr} wrong: {wrong}")