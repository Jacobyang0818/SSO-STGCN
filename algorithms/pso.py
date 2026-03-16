"""
algorithms/pso.py
=================
Particle Swarm Optimization（粒子群最佳化）。

演算法說明：
    每個粒子（sol）維護：
        - X  : 當前位置
        - V  : 當前速度
        - pX : 個體最佳位置
    速度更新公式：
        V = w * V + c1 * r1 * (pX - X) + c2 * r2 * (gX - X)
    其中 w 為慣性權重，c1 為個體學習係數，c2 為社會學習係數。

使用範例：
    from algorithms.pso import PSO

    optimizer = PSO(
        Ngen=10, Nsol=10,
        save_name="my_exp",
        fitness=my_fitness,
        boundary={"lr": (0.001, 0.1), "batch": [16, 32, 64]},
        direction="minimize",
        w=0.7, c1=0.8, c2=0.9,
    )
    optimizer.run()
"""

import numpy as np
import copy
import time

from .base import BaseOptimizer


class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization。

    Parameters
    ----------
    w : float
        慣性權重，控制前一速度對新速度的影響（預設 0.7）。
    c1 : float
        個體學習係數（cognitive coefficient），預設 0.8。
    c2 : float
        社會學習係數（social coefficient），預設 0.9。
    save_name, Ngen, Nsol, fitness, base_param, boundary, direction
        同 BaseOptimizer。
    """

    def __init__(
        self,
        Ngen: int = 10,
        Nsol: int = 10,
        w: float = 0.7,
        c1: float = 0.8,
        c2: float = 0.9,
        save_name: str = None,
        fitness=None,
        base_param=None,
        boundary=None,
        direction: str = "minimize",
    ):
        super().__init__(Ngen=Ngen, Nsol=Nsol, save_name=save_name,
                         fitness=fitness, base_param=base_param,
                         boundary=boundary, direction=direction)

        self.w  = w    # 慣性權重
        self.c1 = c1   # 個體學習係數
        self.c2 = c2   # 社會學習係數

        # PSO 特有：速度矩陣
        self.V  = np.zeros((Nsol, self.Nvar))

        # 初始化個體最佳為 +inf / -inf（避免 gen0 gBest 誤判）
        self.pF = np.full(Nsol, np.inf if self.flag == 1 else -np.inf)

    # ─────────────────────────────────────────────────────────────────────
    # 主搜索流程
    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        """執行 PSO 搜索，完成後儲存結果。"""
        self._init_ckpt()
        self.timestamp = time.time()

        # Gen 0：初始化所有粒子
        for sol in range(self.Nsol):
            self.INIT(sol)

        # Gen 1 ~ Ngen：粒子速度與位置更新
        for gen in range(1, self.Ngen + 1):
            for sol in range(self.Nsol):
                self.UPDATE(sol, gen)

        self._save_result()
        self.result_summary()

    def resume_run(self, log_file: str):
        """從中斷點繼續搜索。"""
        with open(log_file, "rb") as f:
            import pickle
            self.ckpt = pickle.load(f)

        gen, sol = self._load_ckpt_state(self.ckpt)

        if sol == self.Nsol - 1:
            gen, sol = gen + 1, 0
        else:
            sol += 1

        print("=" * 50)
        print(f"🔄 Resume Optimization on Gen: {gen} | Sol: {sol}")
        print("=" * 50)

        self.resume_flag = True
        self.timestamp = time.time()

        if gen == 0:
            for s in range(sol, self.Nsol):
                self.INIT(s)
        elif sol != 0:
            for s in range(sol, self.Nsol):
                self.UPDATE(s, gen)
            gen += 1

        for g in range(gen, self.Ngen + 1):
            for s in range(self.Nsol):
                self.UPDATE(s, g)

        self._save_result()
        self.result_summary()

    # ─────────────────────────────────────────────────────────────────────
    # 核心演算法
    # ─────────────────────────────────────────────────────────────────────

    def INIT(self, sol: int):
        """初始化第 sol 個粒子（速度設為 0，位置隨機或使用基準值）。"""
        if sol == 0 and self.base_param is not None:
            filtered = {k: v for k, v in self.base_param.items() if k in self.boundary}
            self.X[sol] = np.array(list(filtered.values()))
        else:
            self.X[sol] = self.generate_random_sol()

        self.F[sol], record_message = self.fitness(self.X[sol])
        self.pX[sol] = copy.deepcopy(self.X[sol])
        self.pF[sol] = self.F[sol]

        if self.F[sol] * self.flag < self.pF[self.gBest] * self.flag:
            self.gBest = sol

        self.search_time += time.time() - self.timestamp
        self.ckpt_update(0, sol, record_message)

    def UPDATE(self, sol: int, gen: int):
        """
        PSO 速度與位置更新。
        
        速度公式：
            V = w * V + c1 * r1 * (pX - X) + c2 * r2 * (gX - X)
        位置公式：
            X_new = X + V  （clip 至邊界）
        """
        r1 = np.random.rand(self.Nvar)  # 個體學習隨機係數
        r2 = np.random.rand(self.Nvar)  # 社會學習隨機係數

        inertia   = self.w  * self.V[sol]
        cognitive = self.c1 * r1 * (self.pX[sol]          - self.X[sol])
        social    = self.c2 * r2 * (self.pX[self.gBest]   - self.X[sol])

        self.V[sol] = inertia + cognitive + social
        tmp_x = np.clip(self.X[sol] + self.V[sol], self.lower_bound, self.upper_bound)

        self.F[sol], record_message = self.fitness(tmp_x)

        # 僅在 fitness != 0 時接受新位置
        if self.F[sol] != 0:
            self.X[sol] = tmp_x

        # 更新個體最佳
        if self.F[sol] * self.flag < self.pF[sol] * self.flag:
            self.pF[sol] = self.F[sol]
            self.pX[sol] = copy.deepcopy(self.X[sol])
            # 更新全局最佳
            if self.F[sol] * self.flag <= self.pF[self.gBest] * self.flag:
                self.gBest   = sol
                self.genBest = gen

        self.search_time += time.time() - self.timestamp
        self.ckpt_update(gen, sol, record_message)
