import collections
import copy
from collections.abc import Iterable
from typing import List
import numpy as np
import pandas as pd
import cvxpy as cp

# Базовый класс для выпуклой оптимизации портфеля с cvxpy

class Optimizer():
    def __init__(
        self,
        expected_returns,
        cov_matrix,
    ):  
        self.cov_matrix = cov_matrix
        self.expected_returns = expected_returns
        self._max_return_value = None

        num_assets = len(expected_returns)

        tickers = list(expected_returns.index)
        self.tickers = tickers

        self.n_assets = num_assets
        self._w = cp.Variable(num_assets)
        # цель
        self._objective = None
        # ограничения
        self._constraints = []
        # служ.
        self._opt = None
        self._solver = None
        
        # ограничение каждой доли от 0 до 1
        lower_bounds = np.array([0] * self.n_assets)
        upper_bounds = np.array([1] * self.n_assets)
        self.add_constraint(lambda w: w >= lower_bounds)
        self.add_constraint(lambda w: w <= upper_bounds)
        
    # Округление и обнуление близких к нулю долей
    def clean_weights(self, cutoff=0.0001, rounding=5):
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)
    
    # обьединить веса с тикерами для читабельного вывода
    def _make_output_weights(self, weights=None):
        if weights is None:
            weights = self.weights
        return collections.OrderedDict(zip(self.tickers, weights))

    # Начать решение задачи выпуклой оптимизации
    def _solve_cvxpy_opt_problem(self):
        #  создание проблемы из целей и ограничений
        self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self._opt.solve(
            solver=self._solver, verbose=False
        )
        self.weights = self._w.value.round(16) #+ 0.0
        return self._make_output_weights()

    # добавить ограничение для оптимизатора
    def add_constraint(self, new_constraint):
        self._constraints.append(new_constraint(self._w))

    # Оптимизация с нахождением минимального риска
    def min_volatility(self):
        self._objective = portfolio_variance(
            self._w, self.cov_matrix
        )
        # ограничение суммы весов 
        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()
    
    # Оптимизация с нахождение максимальной доходности (используется только внутри других оптимизатров, которые мы используем например effecient_risk)
    def _max_return(self):
        self._objective = portfolio_return(
            self._w, self.expected_returns
        )

        # ограничение суммы весов 
        self.add_constraint(lambda w: cp.sum(w) == 1)

        res = self._solve_cvxpy_opt_problem()

        return -self._opt.value

    # Максимизация прибыли при заданном риске
    def efficient_risk(self, target_volatility):
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError("Заданный риск должен быть float и >= 0")
 
        global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.cov_matrix)))

        if target_volatility < global_min_volatility:
            raise ValueError("Минимальный риск равен {:.3f}. Используйте более высокий заданный риск".format(global_min_volatility))

        self._objective = portfolio_return(
            self._w, self.expected_returns
        )
        variance = portfolio_variance(self._w, self.cov_matrix)

        target_variance = cp.Parameter(
            name="target_variance", value=target_volatility**2, nonneg=True
        )
        self.add_constraint(lambda _: variance <= target_variance)
        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()
    
    # Минимизация риска при заданной доходности
    def efficient_return(self, target_return):
        if not isinstance(target_return, float):
            raise ValueError("Заданная доходность должна быть float")
        if not self._max_return_value:
            a = self.deepcopy()
            # Найти максимальную доходность
            self._max_return_value = a._max_return()
        if target_return > self._max_return_value:
            raise ValueError("Заданная доходность должна быть меньше чем максимально возможная")

        self._objective = portfolio_variance(
            self._w, self.cov_matrix
        )
        ret = portfolio_return(
            self._w, self.expected_returns, negative=False
        )

        target_return_par = cp.Parameter(name="target_return", value=target_return)
        self.add_constraint(lambda _: ret >= target_return_par)
        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    # копия для доп расчета доходности (149, по другому cvxpy не работает типо)
    def deepcopy(self):
        self_copy = copy.copy(self)
        self_copy._constraints = [copy.copy(con) for con in self_copy._constraints]
        return self_copy
    
    # Получить доходность и риск портфеля
    def portfolio_performance(self):
        new_weights = np.asarray(self.weights)

        # Риск (станд откл)
        sigma = np.sqrt(portfolio_variance(new_weights, self.cov_matrix))

        if self.expected_returns is not None:
            mu = portfolio_return(new_weights, self.expected_returns, negative=False)
            return mu, sigma
        else:
            return None, sigma   

# Получить значение (универсально)
def _objective_value(w, obj):
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj

def portfolio_variance(w, cov_matrix):
    variance = cp.quad_form(w, cov_matrix)
    return _objective_value(w, variance)

def portfolio_return(w, expected_returns, negative=True):
    sign = -1 if negative else 1
    # Матричное перемножение
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)

