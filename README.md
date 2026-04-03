# BTC Prediction

## Run

From the project folder (where `run.py` lives):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## Change epochs

Open `btc_predict/config.py` and edit the `Config` class:
- **`torch_epochs`** — training epochs for the normal pipeline run
- **`epoch_sweep_values`** — tuple of epoch counts used when the epoch sweep runs

Set **`run_epoch_sweep`** to `False` if you only want the single run with `torch_epochs` and no sweep

*Download the dataset from below kaggle link and place into a data/raw/ (need to create this in the home directory) directory https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
