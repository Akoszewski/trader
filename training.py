import cma
import numpy as np
from trader import StrategyTester, weightedMajorEmasStrategy, daysToIntervals, readData
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

def train_cmaes(params0, dataTrain, dataTest, chunkSize, startDelay):
    silenceLevel = 2

    def objectiveTrain(params):
        return -StrategyTester.testStrategy(20, weightedMajorEmasStrategy, dataTrain, params, chunkSize, startDelay, silenceLevel)
    def objectiveTest(params):
        return -StrategyTester.testStrategy(20, weightedMajorEmasStrategy, dataTest, params, chunkSize, startDelay, silenceLevel)

    executor = ThreadPool(cpu_count())

    es = cma.CMAEvolutionStrategy(params0, 0.5, {'verbose': 1, 'popsize': 100, 'ftarget': -3.0 })
    while not es.stop():
        solutions = es.ask()
        results = executor.map(objectiveTrain, solutions)
        es.tell(solutions, results)
        trainResult, testResult = es.result[1], objectiveTest(es.result[0])
        print(f'trainResult={trainResult}, testResult={testResult}')
        es.logger.add()  # write data to disc to be plotted
        es.disp(20)      # display info every 20th iteration
    
    # final output
    print('termination by', es.stop())
    print('best f-value =', es.result[1])
    print('best solution =', es.result[0])
    es.result_pretty()
    cma.plot()           # shortcut for es.logger.plot()
    cma.s.figsave('plot.png')
    return solutions

def main():
    paramsSafest = [0.3, -0.3, 0.45134, 0.6169, 0.39278, 0.788256]
    trainTestSplit = 0.6
    dataTrain = readData("./data/hourly/eth.csv", splitRatio=trainTestSplit)
    dataTest = readData("./data/hourly/eth.csv", splitRatio=trainTestSplit, isTest=True)
    chunkSize = 300 # daysToIntervals(300)
    startDelay = 200 # daysToIntervals(200)
    parameters = train_cmaes(paramsSafest, dataTrain, dataTest, chunkSize, startDelay)
    print(parameters)

if __name__ == '__main__':
    main()
