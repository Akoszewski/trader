data = []
with open("./data/hourly/EURUSD60.csv", "r") as f:
    rawData = f.readlines()
    data = []
    for row in rawData:
        row = row.replace(" ", ",")
        row = row.replace("\t", ",")
        data.append(row)

with open("./data/hourly/EURUSD60-done.csv", "w") as f:
    for row in data:
        f.write(row)
