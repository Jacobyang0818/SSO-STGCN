"""
algorithms/sso.py
=================
Social Spider Optimization（社會蜘蛛最佳化）。

演算法說明：
    每次迭代中，對每個解的每一個維度，依隨機數 r 決定更新來源：
        r < Cg  → 採用全局最佳解（gBest）的值
        Cg ≤ r < Cp → 採用個體最佳解（pBest）的值
        r > Cw  → 採用隨機解的值
    （三個條件不互斥，Cg < Cp ≤ Cw）

使用範例：
    from algorithms.sso import SSO

    def my_fitness(x):
        result = ...          # 執行訓練並回傳 metric
        message = {...}       # 自訂紀錄資訊
        return fitness_value, message

    optimizer = SSO(
        Ngen=10, Nsol=10,
        save_name="my_exp",
        fitness=my_fitness,
        boundary={"lr": (0.001, 0.1), "batch": [16, 32, 64]},
        direction="minimize",
    )
    optimizer.run()
    print(optimizer.best_params)
"""

import numpy as np
import copy
import time

from .base import BaseOptimizer


class SSO(BaseOptimizer):
    """
    Social Spider Optimization。

    Parameters
    ----------
    Ngen : int
        迭代代數。
    Nsol : int
        每代的解數量。
    Cg : float
        全局最佳解更新機率閾值（0 < Cg < Cp < Cw < 1）。
    Cp : float
        個體最佳解更新機率閾值。
    Cw : float
        隨機解更新機率閾值。
    save_name, fitness, base_param, boundary, direction
        同 BaseOptimizer。
    """

    def __init__(
        self,
        Ngen: int = 10,
        Nsol: int = 10,
        Cg: float = 0.7,
        Cp: float = 0.8,
        Cw: float = 0.9,
        save_name: str = None,
        fitness=None,
        base_param=None,
        boundary=None,
        direction: str = "minimize",
    ):
        super().__init__(Ngen=Ngen, Nsol=Nsol, save_name=save_name,
                         fitness=fitness, base_param=base_param,
                         boundary=boundary, direction=direction)

        # SSO 特有的更新參數
        self.Cg = Cg   # 全局最佳吸引力（[0, Cg) 採 gBest）
        self.Cp = Cp   # 個體最佳吸引力（[Cg, Cp) 採 pBest）
        self.Cw = Cw   # 隨機擾動閾值（(Cw, 1] 隨機更新）

    # ─────────────────────────────────────────────────────────────────────
    # 主搜索流程
    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        """執行 SSO 搜索，完成後儲存結果。"""
        self._init_ckpt()
        self.timestamp = time.time()

        # Gen 0：初始化所有解
        for sol in range(self.Nsol):
            self.INIT(sol)

        # Gen 1 ~ Ngen：迭代更新
        for gen in range(1, self.Ngen + 1):
            for sol in range(self.Nsol):
                self.UPDATE(sol, gen)

        self._save_result()
        self.result_summary()

    def resume_run(self, log_file: str):
        """
        從中斷點繼續搜索。

        Parameters
        ----------
        log_file : str
            暫存 log.pkl 的路徑（通常為 sso_checkpoint/log.pkl）。
        """
        with open(log_file, "rb") as f:
            import pickle
            self.ckpt = pickle.load(f)

        gen, sol = self._load_ckpt_state(self.ckpt)

        # 計算下一個執行起點
        if sol == self.Nsol - 1:
            gen, sol = gen + 1, 0
        else:
            sol += 1

        print("=" * 50)
        print(f"🔄 Resume Optimization on Gen: {gen} | Sol: {sol}")
        print("=" * 50)

        self.resume_flag = True
        self.timestamp = time.time()

        # 補完初始化
        if gen == 0:
            for s in range(sol, self.Nsol):
                self.INIT(s)
        elif sol != 0:
            for s in range(sol, self.Nsol):
                self.UPDATE(s, gen)
            gen += 1

        # 繼續剩下的代
        for g in range(gen, self.Ngen + 1):
            for s in range(self.Nsol):
                self.UPDATE(s, g)

        self._save_result()
        self.result_summary()

    # ─────────────────────────────────────────────────────────────────────
    # 核心演算法
    # ─────────────────────────────────────────────────────────────────────

    def INIT(self, sol: int):
        """
        初始化第 sol 個解並計算適應度。
        若 sol==0 且提供 base_param，則使用基準值；否則隨機產生。
        """
        if sol == 0 and self.base_param is not None:
            filtered = {k: v for k, v in self.base_param.items() if k in self.boundary}
            self.X[sol] = np.array(list(filtered.values()))
        else:
            self.X[sol] = self.generate_random_sol()

        self.F[sol], record_message = self.fitness(self.X[sol])

        # Gen 0 時個體最佳就是自身
        self.pX[sol] = copy.deepcopy(self.X[sol])
        self.pF[sol] = self.F[sol]

        # 更新全局最佳
        if self.F[sol] * self.flag < self.F[self.gBest] * self.flag:
            self.gBest = sol

        self.search_time += time.time() - self.timestamp
        self.ckpt_update(gen=0, sol=sol, record_message=record_message)

    def UPDATE(self, sol: int, gen: int):
        """
        SSO 更新規則：依隨機遮罩決定每個維度的更新來源。
        為避免修改原始解向量（X），先在 tmp_x 上操作，fitness 計算後才寫入。
        """
        rnd   = np.random.rand(self.Nvar)    # 每個維度的隨機數
        rnd_sol = self.generate_random_sol() # 全隨機解（隨機擾動用）

        # 依機率閾值決定更新來源
        mask_g  = rnd < self.Cg                           # 採 gBest
        mask_p  = (rnd >= self.Cg) & (rnd < self.Cp)     # 採 pBest
        mask_rnd = rnd > self.Cw                          # 採 random

        tmp_x = copy.deepcopy(self.X[sol])
        tmp_x[mask_g]   = self.pX[self.gBest, mask_g]     # gBest 更新
        tmp_x[mask_p]   = self.pX[sol, mask_p]            # pBest 更新
        tmp_x[mask_rnd] = rnd_sol[mask_rnd]               # random 更新

        self.F[sol], record_message = self.fitness(tmp_x)

        # 僅在 fitness != 0 時才接受新解（避免訓練失敗的無效解）
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
        self.ckpt_update(gen=gen, sol=sol, record_message=record_message)
