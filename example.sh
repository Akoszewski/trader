 source .venv/bin/activate
 python forecast_nn.py --file ./data/hourly/eth.csv --horizon-hours 96 --window-hours 168 --target-return 0.02 --probability-threshold 0.52 --hidden-size 64