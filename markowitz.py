
import numpy as np
import pandas as pd
from optimizer import Optimizer
from moex.moex import load_historical
from json import load

def read_config(path):
    cfg = open(path, 'r')
    return load(cfg)

# Загрузка конфига
cfg = read_config('config.json')

# Получение данных по ценам акций
def getStocksData(start, end):
    global cfg
    df = load_historical(cfg.get('tickers'), start, end)
    return df

# Получение минимального дохода
def getMinReturn(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma,) 
    ef.efficient_return(float(0))
    return ef.portfolio_performance()[0]

# Получение максимально возможного риска портфеля
def getMaxRisk(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma) 
    ef.efficient_risk(float(1))
    return ef.portfolio_performance()[1]

# Годовая доходность
def get_mu(prices):
    frequency = 251
    returns = prices.pct_change().dropna(how="all")
    return (1 + returns).prod() ** (frequency / returns.count()) - 1

# Cov
def get_sigma(prices):
    frequency = 251
    returns = prices.pct_change().dropna(how="all")
    return returns.cov() * frequency

# Получение минимально возможного риска портфеля
def getMinRisk(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma) 
    ef.min_volatility()
    return ef.portfolio_performance()[1]

# Получение максимально возможного риска портфеля
def getMaxReturn(stocks):
    return max(get_mu(stocks).values) - 0.01

# Минимальный риск при заданной доходности
def minimize_risk(stocks, target_return: float):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma)
    minrisk=ef.efficient_return(target_return)
    return ef

# Максимальная доходность для заданного риска
def maximize_return(stocks, target_risk: float):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma) 
    maxret=ef.efficient_risk(target_risk)
    return ef

# Получение истории поведения портфеля на истории
def getPortfolioHistory(deposit, weights, stocks):
    amounts = dict()
    # получение количества акций
    for w in weights:
        price = stocks[w][0]
        amount = (deposit * weights[w]) / price
        amounts[w] = amount
    value = []
    dates = []
    # расчет во времени
    for date, s in stocks.iterrows():
        cost = 0
        for a in amounts:
            cost += s[a] * amounts[a]
        dates.append(date)
        value.append(cost)
    perf = pd.DataFrame({'value': value, 'date': dates})
    return perf