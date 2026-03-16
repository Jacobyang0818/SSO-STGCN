import sys
import os
import glob
from algorithms import SSO, GA, PSO

# 模擬一個具有明確最小值的假適應度函數
def dummy_fitness(x):
    # Sphere function: f(x) = sum(x_i^2), 最佳解在 [0, 0, ...]
    fitness_val = sum(i**2 for i in x)
    msg = {'info': 'dummy test passed'}
    return fitness_val, msg

boundary = {
    'x1': (-10.0, 10.0),
    'x2': (-10.0, 10.0),
    'x3': [-5, 0, 5]
}

def test_alg(name, optimizer_cls, **kwargs):
    print(f"\n{'='*50}\nTesting {name}...\n{'='*50}")
    
    # 清理舊的結果
    for p in glob.glob(f"sso_result/{name}_test_*.pkl"):
        os.remove(p)
        
    optimizer = optimizer_cls(
        Ngen=3, Nsol=3,
        save_name=f"{name}_test",
        fitness=dummy_fitness,
        boundary=boundary,
        direction="minimize",
        **kwargs
    )
    
    # 執行搜索
    optimizer.run()
    
    # 驗證輸出檔案是否存在
    output_files = glob.glob(f"sso_result/{name}_test_*.pkl")
    if not output_files:
        print(f"❌ {name} failed to save result file in sso_result/!")
        return False
        
    print(f"✅ {name} saved result file: {output_files[0]}")
    
    # 測試讀取功能
    print(f"\n[Testing load_result for {name}]")
    optimizer2 = optimizer_cls(
        Ngen=3, Nsol=3, save_name=f"{name}_test_load",
        fitness=dummy_fitness, boundary=boundary
    )
    optimizer2.load_result(output_files[0], print_message=True)
    if optimizer2.best_params is not None:
        print(f"✅ {name} load_result successful. Best params: {optimizer2.best_params}")
    else:
        print(f"❌ {name} load_result failed to reconstruct best_params!")
        return False
        
    return True

if __name__ == '__main__':
    # 測試確保三個演算法都能正確跑完並存取 .pkl
    sso_ok = test_alg("SSO", SSO)
    ga_ok = test_alg("GA", GA, crossover_rate=0.9, mutation_rate=0.1)
    pso_ok = test_alg("PSO", PSO, w=0.7, c1=0.8, c2=0.9)
    
    if sso_ok and ga_ok and pso_ok:
        print("\n🎉 All 3 algorithms tested successfully!")
    else:
        print("\n⚠️ Some algorithms failed the test.")
