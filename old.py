# tylko jak sie przetna
# def movingAveragesCrossingStrategy(startBalance, x):
#     totalValues = []
#     simulation = Simulation(startBalance)
#     previous200WasOver = True
#     for i in range(200, len(data)):
#         prices = [o.close for o in data]
#         avg50 = average(prices[i-50:i-1])
#         avg200 = average(prices[i-200:i-1])
#         if (avg50 > avg200 and previous200WasOver == True):
#             simulation.buy(x, data[i].open)
#             previous200WasOver = False
#         elif (avg50 < avg200 and previous200WasOver == False):
#             simulation.sell(x, data[i].open)
#             previous200WasOver = True
#         totalValues.append(simulation.totalValue(data[i].open))
#     return totalValues