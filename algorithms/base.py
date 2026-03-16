"""
algorithms/base.py
==================
所有啟發式演算法的抽象基類（Abstract Base Class）。

將 SSO / GA / PSO 三者共用的功能集中在此：
  - Checkpoint 結構定義與更新（ckpt_update）
  - Log 輸出與存檔（print_log）
  - 結果摘要顯示（result_summary）
  - 搜索結果儲存（_save_result）
  - 從中斷點恢復的狀態重建（_load_ckpt_state）
  - 收斂圖繪製（plot）
  - 結果讀取（load_result）
  - 敏感度分析（sensitivity_analysis）

子類別（SSO / GA / PSO）只需實作：
  - __init__：演算法特有的超參數初始化
  - run：主搜索流程
  - INIT：解的初始化
  - UPDATE：解的更新
  - generate_random_sol：隨機解產生
"""

import numpy as np
import time, pickle, copy, os, uuid
from typing import Callable, Dict, Optional, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


class BaseOptimizer(ABC):
    """
    啟發式演算法抽象基類。

    Parameters
    ----------
    Ngen : int
        迭代代數（generation 數量）。
    Nsol : int
        每代的解數量（particle / individual 數量）。
    save_name : str
        結果檔案的前綴名稱，用於命名輸出的 .pkl 檔。
    fitness : Callable
        適應度函數，接受一個解向量並回傳 (fitness_value, record_message)。
    base_param : dict, optional
        初始解的基準參數，gen0/sol0 會使用此值而非隨機產生。
    boundary : dict
        搜索空間定義，格式為 {param_name: (lower, upper)} 或 {param_name: [v1, v2, ...]}.
    direction : str
        最佳化方向，'minimize'（最小化）或 'maximize'（最大化）。
    """

    # 儲存/log 的固定路徑設定
    LOG_DIR  = "sso_checkpoint/"     # 暫存 checkpoint 的目錄（中斷恢復用）
    SAVE_DIR = "sso_result/"         # 最終搜索結果目錄

    def __init__(
        self,
        Ngen: int = 10,
        Nsol: int = 10,
        save_name: str = None,
        fitness: Callable = None,
        base_param: Optional[Dict] = None,
        boundary: Dict[str, Union[tuple, list]] = None,
        direction: str = "minimize",
    ):
        # ── 演算法基本設定 ──────────────────────────────────────────────
        self.Ngen = Ngen
        self.Nsol = Nsol
        self.fitness = fitness
        self.base_param = base_param
        self.boundary = boundary
        self.save_name = save_name

        # 驗證 boundary 不為空
        assert boundary is not None, "boundary 必須提供，不可為 None"

        # ── 解向量維度與邊界 ────────────────────────────────────────────
        self.Nvar = len(boundary)
        self.lower_bound = np.array([
            b[0] if isinstance(b, tuple) else min(b)
            for b in boundary.values()
        ], dtype=float)
        self.upper_bound = np.array([
            b[1] if isinstance(b, tuple) else max(b)
            for b in boundary.values()
        ], dtype=float)

        # ── 解矩陣（由子類別補充 X, V 等） ─────────────────────────────
        self.X  = np.zeros((Nsol, self.Nvar))   # 當前解
        self.pX = np.zeros((Nsol, self.Nvar))   # 個體最佳解
        self.F  = np.zeros(Nsol, dtype=float)   # 當前適應度
        self.pF = np.zeros(Nsol, dtype=float)   # 個體最佳適應度

        # ── 最佳解追蹤 ─────────────────────────────────────────────────
        self.gBest   = 0      # 全局最佳解的 sol index
        self.genBest = 0      # 全局最佳解出現的 gen index
        self.best_params = None

        # ── 方向設定（最小化 / 最大化） ─────────────────────────────────
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction 必須是 'minimize' 或 'maximize'")
        self.direction = direction
        self.flag = 1 if direction == "minimize" else -1  # flag=1: 最小化；flag=-1: 最大化

        # ── Checkpoint 與日誌 ────────────────────────────────────────────
        self.ckpt = {}                                      # 完整搜索紀錄（巢狀 dict）
        self.log_dir  = self.LOG_DIR
        self.log_path = os.path.join(self.log_dir, "log.pkl")  # 暫存 log（中斷恢復用）
        self.save_path = None                               # 最終結果 .pkl 路徑

        # ── 計時器 ──────────────────────────────────────────────────────
        self.timestamp   = None
        self.search_time = 0.0
        self.resume_flag = None

    # ─────────────────────────────────────────────────────────────────────
    # 抽象方法（子類別必須實作）
    # ─────────────────────────────────────────────────────────────────────

    @abstractmethod
    def run(self):
        """執行完整搜索流程。"""
        ...

    @abstractmethod
    def INIT(self, sol: int):
        """初始化第 sol 個解。"""
        ...

    @abstractmethod
    def UPDATE(self, *args, **kwargs):
        """更新解（各演算法更新規則不同）。"""
        ...

    @abstractmethod
    def generate_random_sol(self) -> np.ndarray:
        """在搜索空間內生成一個隨機解。"""
        ...

    # ─────────────────────────────────────────────────────────────────────
    # 共用工具方法
    # ─────────────────────────────────────────────────────────────────────

    def _init_ckpt(self):
        """初始化 ckpt 字典結構，並確保 log 目錄存在。"""
        os.makedirs(self.log_dir, exist_ok=True)
        for gen in range(self.Ngen + 1):
            self.ckpt.setdefault(gen, {})
        # 建立空 log 檔（用於中斷恢復）
        if not os.path.exists(self.log_path):
            with open(self.log_path, "wb"):
                pass

    def _save_result(self):
        """
        將搜索結果存為 .pkl 檔，並刪除暫存 log。
        檔名格式：{save_name}_ggen{genBest}_gsol{gBest}_searchtime{t}_{uid}.pkl
        """
        self.best_params = copy.deepcopy(self.pX[self.gBest])
        uid = uuid.uuid4().hex[:8]
        self.save_path = (
            f"{self.SAVE_DIR}{self.save_name}"
            f"_ggen{self.genBest}_gsol{self.gBest}"
            f"_searchtime{self.search_time}_{uid}.pkl"
        )
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(self.ckpt, f)
        print(f"📄 記錄檔已儲存至：{self.save_path}")
        print("-" * 40)

        # 搜索完成後刪除暫存 log
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def ckpt_update(self, gen: int, sol: int, record_message: dict):
        """
        更新 ckpt 字典並即時儲存至 log.pkl。

        ckpt 結構：
            ckpt[gen][sol] = {
                'x'           : (當前解, 當前適應度),
                'p'           : (個體最佳解, 個體最佳適應度),
                'g'           : (全局最佳解, 全局最佳適應度, genBest, gBest),
                'search_time' : 累計搜索時間（秒）,
                'message'     : fitness 函數回傳的自訂紀錄字典,
            }
        """
        solution = tuple(self.X[sol].copy())
        self.ckpt[gen][sol] = {
            "x"           : (solution, self.F[sol]),
            "p"           : (tuple(self.pX[sol].copy()), self.pF[sol]),
            "g"           : (tuple(self.pX[self.gBest].copy()), self.pF[self.gBest], self.genBest, self.gBest),
            "search_time" : self.search_time,
            "message"     : record_message,
        }
        log_msg = (
            f"Gen : {gen:>3} | Sol : {sol:>3} | "
            f"fitness : {self.F[sol]:>20.6f} | "
            f"search_time : {self.search_time:>20.6f} | "
            f"params : {solution} | "
            f"record_message : {record_message}"
        )
        self.print_log(log_msg)

    def print_log(self, msg: str, print_time: bool = True):
        """
        印出 log 訊息並即時存檔至 log.pkl（用於中斷恢復）。

        Parameters
        ----------
        msg : str
            要印出的訊息。
        print_time : bool
            是否在訊息前加上本機時間戳記。
        """
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            log = f"[ {localtime} ] {msg}"
        else:
            log = msg
        print(log)

        # 即時儲存 ckpt 至暫存 log，確保中斷後可恢復
        with open(self.log_path, "wb") as f:
            pickle.dump(self.ckpt, f)

        # 重設計時起點（避免 I/O 期間的時間被計入搜索時間）
        self.timestamp = time.time()

    def result_summary(self):
        """印出搜索結果摘要。"""
        print("=" * 40)
        print("🚀 Search Results Summary")
        print("=" * 40)
        print(f"✅ Search 完成！")
        print(f"🔍 最佳解所在世代編號：{self.genBest}")
        print(f"🔍 最佳解索引編號：{self.gBest}")
        print("-" * 40)
        print("🏆 Optimal Solution Details:")
        print(f"  - Optimal Solution: {self.ckpt[self.genBest][self.gBest]}")
        print(f"  - Optimal Fitness Value: {self.pF[self.gBest]:.6f}")
        print("-" * 40)
        if self.resume_flag:
            print("⚠️  這是接續優化的時間，請自行補上之前的運行時間")
        print(f"⏱️  Total Search Time: {self.search_time:.2f} seconds")
        print("=" * 40)

    def generate_random_sol(self) -> np.ndarray:
        """
        根據 boundary 定義生成一個隨機解向量。

        支援的邊界型別：
            - tuple (int, int)   → 整數均勻分佈
            - tuple (float, float) → 連續均勻分佈
            - list [v1, v2, ...] → 從列表中隨機選取
        """
        sol = []
        for key, b in self.boundary.items():
            if isinstance(b, tuple):
                lo, hi = b
                if isinstance(lo, int) and isinstance(hi, int):
                    sol.append(int(np.random.randint(lo, hi + 1)))
                else:
                    sol.append(float(np.random.uniform(lo, hi)))
            elif isinstance(b, list):
                sol.append(np.random.choice(b))
            else:
                raise ValueError(f"不支援的 boundary 型別，參數 '{key}': {b}")
        return np.array(sol)

    def _load_ckpt_state(self, ckpt: dict):
        """
        從 ckpt 字典中重建搜索狀態（供 resume_run 使用）。

        Returns
        -------
        gen : int  恢復點的代數
        sol : int  恢復點的解索引
        """
        valid_gens = [g for g, v in ckpt.items() if v != {}]
        if not valid_gens:
            raise RuntimeError("ckpt 中找不到有效的 gen 紀錄")

        gen = max(valid_gens)
        sol = max(ckpt[gen].keys())

        # 重建全局最佳
        self.genBest     = ckpt[gen][sol]["g"][2]
        self.gBest       = ckpt[gen][sol]["g"][3]
        self.search_time = ckpt[gen][sol]["search_time"]

        # 重建每個解的個體狀態
        for s in range(self.Nsol):
            use_gen = None
            if gen in ckpt and s in ckpt[gen] and ckpt[gen][s]:
                use_gen = gen
            elif (gen - 1) in ckpt and s in ckpt[gen - 1] and ckpt[gen - 1][s]:
                use_gen = gen - 1
            else:
                print(f"[警告] 無法找到第 {s} 個 solution，跳過")
                continue
            self.pX[s] = np.array(ckpt[use_gen][s]["p"][0])
            self.pF[s] = float(ckpt[use_gen][s]["p"][1])
            self.X[s]  = np.array(ckpt[use_gen][s]["x"][0])
            self.F[s]  = float(ckpt[use_gen][s]["x"][1])

        return gen, sol

    def load_result(self, result_file: str, print_message: bool = False):
        """
        讀取已完成的搜索結果（.pkl），並重建 gBest 狀態。

        Parameters
        ----------
        result_file : str
            結果 .pkl 檔路徑。
        print_message : bool
            是否額外印出最佳解的詳細紀錄。
        """
        if not os.path.exists(result_file):
            print("❌ 找不到結果檔案")
            return
        try:
            with open(result_file, "rb") as f:
                self.ckpt = pickle.load(f)
            self._load_ckpt_state(self.ckpt)
            print(f"📂 File ggen: {self.genBest}, gsol: {self.gBest}")
            if print_message:
                entry = self.ckpt.get(self.genBest, {}).get(self.gBest)
                if entry:
                    self.best_params = np.array(copy.deepcopy(entry["x"][0]))
                    print("=" * 40, "\n📝 Record Information\n" + "=" * 40)
                    print(f"🔍 Record Message: {entry['message']}")
                    print(f"🔧 Parameters: {self.best_params}")
                    print(f"🏆 Fitness: {entry['x'][1]}\n" + "=" * 40)
                else:
                    print("❌ Record not found in checkpoint!")
        except Exception as e:
            print(f"⚠️ Error while reading checkpoint: {e}")

    def plot(self):
        """
        繪製搜索過程的適應度收斂圖。
        產生兩張圖：
            1. 每個解（sol）的個體最佳值（pbest）隨代數變化
            2. 全局最佳值（gbest）隨代數變化
        """
        plt.rcParams.update({"font.size": 18})

        # 整理 ckpt 資料為 DataFrame
        data = []
        for gen, sol_data in self.ckpt.items():
            for sol, values in sol_data.items():
                data.append({
                    "gen"  : gen,
                    "sol"  : sol,
                    "pbest": values["p"][1],
                    "gbest": values["g"][1],
                })
        df = pd.DataFrame(data)

        # 圖一：各 sol 的 pbest 收斂曲線
        plt.figure(figsize=(10, 6))
        for sol, group in df.groupby("sol"):
            plt.plot(group["gen"], group["pbest"], marker="o", label=f"sol {sol}")
        plt.title("Fitness Value Evolution per Solution")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (pbest)")
        plt.grid(alpha=0.3)
        plt.legend(title="sol")
        plt.xticks(sorted(df["gen"].unique()))
        plt.tight_layout()
        plt.show()

        # 圖二：全局最佳 gbest 收斂曲線
        if self.flag == 1:
            gbest_df = df.groupby("gen")["gbest"].min().reset_index()
        else:
            gbest_df = df.groupby("gen")["gbest"].max().reset_index()
        plt.figure(figsize=(8, 6))
        plt.plot(gbest_df["gen"], gbest_df["gbest"], marker="o", color="red", label="Global Best")
        plt.title("Global Best Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (gbest)")
        plt.xticks(sorted(df["gen"].unique()))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sensitivity_analysis(self, files: list, label_map: dict = None):
        """
        對多個搜索結果 .pkl 檔進行平行座標（Parallel Coordinates）敏感度分析。

        Parameters
        ----------
        files : list of str
            搜索結果 .pkl 檔路徑列表。
        label_map : dict, optional
            將離散型參數的原始值映射為可讀標籤，格式：
            {param_name: [label_v1, label_v2, ...]}.
        """
        param_names = list(self.boundary.keys())
        param_data, fitness_data = [], []

        for fp in files:
            self.load_result(fp)
            for gen_data in self.ckpt.values():
                for sol_data in gen_data.values():
                    param_data.append(sol_data["x"][0])
                    fitness_data.append(sol_data["x"][1])

        param_df = pd.DataFrame(param_data, columns=param_names)
        param_df["fitness"] = fitness_data

        value_to_label_map = {}
        dimension_ranges   = {}

        if label_map:
            for param, labels in label_map.items():
                original_values = self.boundary[param]
                uniform_indices = list(range(len(labels)))
                param_df[param] = param_df[param].map(dict(zip(original_values, uniform_indices)))
                value_to_label_map[param] = dict(zip(original_values, labels))
                dimension_ranges[param]   = [0, len(labels) - 1]

        for param in self.boundary:
            if param not in dimension_ranges:
                b = self.boundary[param]
                if isinstance(b, tuple):
                    dimension_ranges[param] = [b[0], b[1]]
                elif isinstance(b, list):
                    dimension_ranges[param] = [min(b), max(b)]

        GROUP_SIZE = 5
        n_groups = (len(param_names) + GROUP_SIZE - 1) // GROUP_SIZE
        for gi in range(n_groups):
            group_params = param_names[gi * GROUP_SIZE : (gi + 1) * GROUP_SIZE]
            fig = px.parallel_coordinates(
                param_df[group_params],
                dimensions=group_params,
                color=param_df["fitness"],
                color_continuous_scale="Viridis" if self.flag == 1 else "Viridis_r",
                title=f"Sensitivity Analysis (Params {gi*GROUP_SIZE+1}~{min((gi+1)*GROUP_SIZE, len(param_names))})",
            )
            fig.update_coloraxes(colorbar_title="Fitness")
            for dim in fig.data[0]["dimensions"]:
                p = dim["label"]
                if p in dimension_ranges:
                    dim["range"] = dimension_ranges[p]
                if p in value_to_label_map:
                    dim["ticktext"] = list(value_to_label_map[p].values())
                    dim["tickvals"] = list(range(len(value_to_label_map[p])))
            fig.show()
