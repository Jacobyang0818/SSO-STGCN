"""
algorithms/ga.py
================
Genetic Algorithm（遺傳演算法）。

演算法說明：
    1. 初始化 Nsol 個個體
    2. 每代執行：
       a. 錦標賽選擇（Tournament Selection, k=2）
       b. 均值交叉（Arithmetic Crossover, alpha=0.5）
       c. 隨機突變（+1 / -1 離散突變）
    3. 回傳全局最佳解

使用範例：
    from algorithms.ga import GA

    optimizer = GA(
        Ngen=10, Nsol=10,
        save_name="my_exp",
        fitness=my_fitness,
        boundary={"lr": (1, 10), "batch": [16, 32, 64]},
        direction="minimize",
        crossover_rate=0.9,
        mutation_rate=0.1,
    )
    optimizer.run()
"""

import numpy as np
import copy
import time

from .base import BaseOptimizer


class GA(BaseOptimizer):
    """
    Genetic Algorithm。

    Parameters
    ----------
    crossover_rate : float
        交叉率，決定個體之間是否進行交叉操作（預設 0.9）。
    mutation_rate : float
        突變率，決定每個維度是否進行突變（預設 0.1）。
    save_name, Ngen, Nsol, fitness, base_param, boundary, direction
        同 BaseOptimizer。
    """

    def __init__(
        self,
        Ngen: int = 10,
        Nsol: int = 10,
        save_name: str = None,
        fitness=None,
        base_param=None,
        boundary=None,
        direction: str = "minimize",
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
    ):
        super().__init__(Ngen=Ngen, Nsol=Nsol, save_name=save_name,
                         fitness=fitness, base_param=base_param,
                         boundary=boundary, direction=direction)

        self.crossover_rate = crossover_rate   # 交叉率
        self.mutation_rate  = mutation_rate    # 突變率

    # ─────────────────────────────────────────────────────────────────────
    # 主搜索流程
    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        """執行 GA 搜索，完成後儲存結果。"""
        self._init_ckpt()
        self.timestamp = time.time()

        # Gen 0：初始化所有個體
        for sol in range(self.Nsol):
            self.INIT(sol)

        # Gen 1 ~ Ngen：選擇、交叉、突變、評估
        for gen in range(1, self.Ngen + 1):
            self.UPDATE(gen)

        self._save_result()
        self.result_summary()

    def resume_run(self, log_file: str):
        """從中斷點繼續搜索（僅支援從完整代的邊界恢復）。"""
        with open(log_file, "rb") as f:
            import pickle
            self.ckpt = pickle.load(f)

        gen, _ = self._load_ckpt_state(self.ckpt)
        gen += 1  # 從下一代開始

        print("=" * 50)
        print(f"🔄 Resume Optimization from Gen: {gen}")
        print("=" * 50)

        self.resume_flag = True
        self.timestamp = time.time()

        for g in range(gen, self.Ngen + 1):
            self.UPDATE(g)

        self._save_result()
        self.result_summary()

    # ─────────────────────────────────────────────────────────────────────
    # 核心演算法
    # ─────────────────────────────────────────────────────────────────────

    def INIT(self, sol: int):
        """初始化第 sol 個個體並評估適應度。"""
        if sol == 0 and self.base_param is not None:
            filtered = {k: v for k, v in self.base_param.items() if k in self.boundary}
            self.X[sol] = np.array(list(filtered.values()))
        else:
            self.X[sol] = self.generate_random_sol()

        self.F[sol], record_message = self.fitness(self.X[sol])
        self.pX[sol] = copy.deepcopy(self.X[sol])
        self.pF[sol] = self.F[sol]

        if self.F[sol] * self.flag < self.F[self.gBest] * self.flag:
            self.gBest = sol

        self.search_time += time.time() - self.timestamp
        self.ckpt_update(0, sol, record_message)

    def UPDATE(self, gen: int):
        """
        GA 一代更新：
            1. 錦標賽選擇 → selected[Nsol]
            2. 兩兩交叉 → offspring[Nsol]
            3. 逐個突變 → offspring[Nsol]
            4. 評估適應度並更新個體/全局最佳
        """
        # (1) 錦標賽選擇
        selected = self._selection()

        # (2) 交叉
        offspring = []
        for i in range(0, self.Nsol, 2):
            if i + 1 < self.Nsol:
                if np.random.rand() < self.crossover_rate:
                    c1, c2 = self._crossover(selected[i], selected[i + 1])
                else:
                    c1, c2 = selected[i].copy(), selected[i + 1].copy()
                offspring.extend([c1, c2])
            else:
                # 處理奇數 Nsol 的情況，最後一個個體直接保留
                offspring.append(selected[i].copy())

        # (3) 突變
        offspring = [self._mutation(child) for child in offspring]

        # (4) 評估並更新
        for i in range(self.Nsol):
            self.X[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)
            self.F[i], record_message = self.fitness(self.X[i])

            if self.F[i] * self.flag < self.pF[i] * self.flag:
                self.pF[i] = self.F[i]
                self.pX[i] = copy.deepcopy(self.X[i])
                if self.F[i] * self.flag <= self.pF[self.gBest] * self.flag:
                    self.gBest   = i
                    self.genBest = gen

            self.search_time += time.time() - self.timestamp
            self.ckpt_update(gen, i, record_message)

    def _selection(self, k: int = 2) -> list:
        """
        錦標賽選擇：從族群中重複抽取 k 個個體，取適應度較好者。

        Parameters
        ----------
        k : int, default=2
            每次錦標賽的參賽個體數。

        Returns
        -------
        list of np.ndarray
            選出的 Nsol 個解向量。
        """
        selected = []
        for _ in range(self.Nsol):
            idx    = np.random.choice(self.Nsol, k, replace=False)
            winner = idx[np.argmin(self.F[idx])] if self.flag == 1 else idx[np.argmax(self.F[idx])]
            selected.append(self.X[winner].copy())
        return selected

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        """
        算術交叉（Arithmetic Crossover, alpha=0.5）。
        產生兩個子代，為雙親的加權平均。

        Parameters
        ----------
        p1, p2 : np.ndarray
            兩個父代解向量。

        Returns
        -------
        (c1, c2) : tuple of np.ndarray
        """
        alpha = 0.5
        c1 = np.round(alpha * p1 + (1 - alpha) * p2).astype(int)
        c2 = np.round(alpha * p2 + (1 - alpha) * p1).astype(int)
        return c1, c2

    def _mutation(self, child: np.ndarray) -> np.ndarray:
        """
        離散突變：以 mutation_rate 的機率對每個維度加 ±1。

        Parameters
        ----------
        child : np.ndarray
            子代解向量。

        Returns
        -------
        np.ndarray
            突變後的子代（已 clip 至邊界）。
        """
        child = child.copy()
        for i in range(self.Nvar):
            if np.random.rand() < self.mutation_rate:
                child[i] += np.random.choice([-1, 1])
        return np.clip(child, self.lower_bound, self.upper_bound).astype(int)
