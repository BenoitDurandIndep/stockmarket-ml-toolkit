# Stock Market ML Toolkit
"Stock Market ML Toolkit" is a Python-based project that focuses on gathering stock market data, calculating financial indicators, and preparing this data for machine learning model training and backtesting. The base initial data and Framework are stored in a SQLite DB. The project uses a variety of data manipulation techniques based on pandas. It also includes functionality for normalizing and sorting the data, as well as splitting it into training and validation sets. The prepared data is then ready to be used in machine learning models, specifically Long Short-Term Memory (LSTM) models implemented in PyTorch (I gave up TensorFlow), for predicting market trends. 
It's a sandbox project I'm still working on and I don't really know where it's going. If you have any questions don't hesitate :)

## Repository File Structure
    dir
    ├── backtest/          # backtest of the strategies with backtrader or my own framework
    ├── dataset_mngr/      # the toolkit : all the lib to help me in the journey
    ├── DB/                # some scripts for the SQLite DB and migration from MariaDB
    ├── dockers/           # dockers images used for TensorFlow training on Windows with GPU
    ├── get_data/          # scripts to get data from brokers API
    ├── models/            # prepare data and train models
    ├── pipes/             # pipelines used once the models are validated
    ├── tests/             # unit tests for the toolkit

## Installation and Setup
Initially dev with Python 3.10.5
1. Clone the repository: `git clone https://github.com/BenoitDurandIndep/stockmarket-ml-toolkit.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Create SQLiteDb with the script create_sqlite_db.sql
4. Have fun

## Usage

Explore the ipynb notebooks to understand how to use it

## Data

The data for this project comes from Yahoo, Alphavantage and QuantDataManager,  more providers in the futur

## ML Training

The machine learning models are trained using SKLearn for non RNN, I tried TensorFlow for LSTM and finally migrated to Pytorch

## Contributing

If you need help using this framework, or if you want to contribute, please create an issue on the <a href="https://github.com/BenoitDurandIndep/stockmarket-ml-toolkit/issues" target="_blank">GitHub repo</a>
You can send me a message on <a href="https://www.linkedin.com/in/bdu8784/" target="_blank">LinkedIn</a>

## License

GPL-3.0 license [License](https://github.com/BenoitDurandIndep/stockmarket-ml-toolkit/blob/main/LICENSE)
