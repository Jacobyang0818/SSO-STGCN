"""
algorithms 模組
=================
提供以下三種啟發式演算法，用於超參數或模型架構搜索（NAS / HPO）：

    - SSO (Social Spider Optimization)  → algorithms.sso.SSO
    - GA  (Genetic Algorithm)           → algorithms.ga.GA
    - PSO (Particle Swarm Optimization) → algorithms.pso.PSO

所有演算法皆繼承自 algorithms.base.BaseOptimizer，
共用的 checkpoint / log / resume / plot / 分析功能集中在 base.py。

使用範例：
    from algorithms.sso import SSO
    from algorithms.ga  import GA
    from algorithms.pso import PSO
"""

from .sso import SSO
from .ga  import GA
from .pso import PSO

__all__ = ["SSO", "GA", "PSO"]
