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
        # if self.crypto >= amountOfMoney/price:
        if True: # mozna shortowac
            self.money += (1-self.provision)*amountOfMoney
            self.crypto -= amountOfMoney/price

    def totalValue(self, price):
        return self.money + self.crypto * price



def allInStrategy(startBalance, x):
    totalValues = []
    simulation = Simulation(startBalance)
    simulation.buy(startBalance, data[0].open)
    for i in range(len(data)):
        totalValues.append(simulation.totalValue(data[i].open))
    return totalValues


def alwaysBuyStrategy(startBalance, x):
    return True

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
        return True
    else:
        return False

def rsiStrategy(pricesSoFar, n):
    rsi = calculateRSI(pricesSoFar, 15)
    if (rsi > 80):
        return False
    elif (rsi < 20):
        return True

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
        shouldBuy = strategy(prices[:i], 15)
        if (shouldBuy):
            simulation.buy(x, data[i].open)
        else:
            simulation.sell(x, data[i].open)
        totalValues.append(simulation.totalValue(data[i].open))
    return totalValues

def predictNext(prices):
    correct = 0
    wrong = 0
    for i in range(15, len(prices) - 1):
        shouldBuy = rsiStrategy(prices[:i], 15)
        if prices[i+1] > prices[i] and shouldBuy:
            correct = correct + 1
        elif prices[i+1] > prices[i] and not shouldBuy:
            wrong = wrong + 1
        elif prices[i+1] < prices[i] and not shouldBuy:
           correct = correct + 1
        else:
            wrong = wrong + 1
    return [correct, wrong]

def simulateAllInEveryDayDecide(startBalance, strategy):
    totalValues = []
    simulation = Simulation(startBalance)
    prices = [o.close for o in data]
    for i in range(15, len(prices)):
        shouldBuy = strategy(prices[:i], 15)
        if (shouldBuy):
            if (simulation.money > 100):
                simulation.buy(simulation.money, data[i].open)
                print(f"Buy. Curr crypto: {simulation.crypto} curr money: {simulation.money}")
        else:
            if (simulation.crypto * data[i].open > 100):
                simulation.sell(simulation.crypto*data[i].open, data[i].open)
                print(f"Sell. Curr crypto: {simulation.crypto} curr money: {simulation.money}")
            
        totalValues.append(simulation.totalValue(data[i].open))
    print(f"Crypto: {simulation.crypto} money: {simulation.money}")
    return totalValues

data = readData("btc_every_day.csv")
data = data[::]
plotData(data)
startBalance = 1000
totalValues = simulateAllInEveryDayDecide(startBalance, randomBuySellStrategy)
print(f"Na koniec masz: {round(totalValues[-1])} dol z zainwestowanych {startBalance}")

plt.plot(totalValues)
plt.show()

# prices = [o.close for o in data]
# [corr, wrong] = simulateAllInEveryDayDecide(prices)
# print(f"correct: {corr} wrong: {wrong}")