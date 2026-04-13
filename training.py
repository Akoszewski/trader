import cma
import numpy as np
from trader import StrategyTester, weightedMajorEmasStrategy, daysToIntervals, readData

def train_cmaes(params0, data, chunkSize, startDelay, provision = 0.0):
    (trainStart, trainEnd), (testStart, testEnd) = StrategyTester.getTrainTestRanges(len(data.closes), startDelay)
    es = cma.CMAEvolutionStrategy(params0, 0.5, {'verb_disp': 1})
    while not es.stop():
        solutions = es.ask()
        es.tell(
            solutions,
            [
                -StrategyTester.testStrategy(
                    20,
                    weightedMajorEmasStrategy,
                    data,
                    provision,
                    params,
                    chunkSize,
                    startDelay,
                    True,
                    trainStart,
                    trainEnd,
                )
                for params in solutions
            ],
        )
        es.logger.add()  # write data to disc to be plotted
        es.disp()
    es.result_pretty()
    cma.plot()  # shortcut for es.logger.plot()
    cma.s.figsave('plot.png')

    rankedSolutions = []
    for params in solutions:
        trainResult = StrategyTester.testStrategy(
            20,
            weightedMajorEmasStrategy,
            data,
            provision,
            params,
            chunkSize,
            startDelay,
            True,
            trainStart,
            trainEnd,
        )
        testResult = StrategyTester.testStrategy(
            20,
            weightedMajorEmasStrategy,
            data,
            provision,
            params,
            chunkSize,
            startDelay,
            True,
            testStart,
            testEnd,
        )
        rankedSolutions.append((trainResult, testResult, params))

    rankedSolutions.sort(reverse = True)
    return rankedSolutions

if __name__ == '__main__':
    paramsSafest = [0.3, -0.3, 0.45134, 0.6169, 0.39278, 0.788256]
    data = readData("./data/hourly/eth.csv")
    data.initTechnicals()
    chunkSize = daysToIntervals(300)
    startDelay = daysToIntervals(200)
    parameters = train_cmaes(paramsSafest, data, chunkSize, startDelay)
    print(parameters)
