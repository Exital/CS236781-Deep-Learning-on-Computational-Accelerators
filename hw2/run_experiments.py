from hw2.experiments import run_experiment
from sys import argv
from time import time
import random

# K = filters per layer
# L = layers per blocks
RESULTS_PATH = "./results"
BATCH_NUM = 200
SEED = random.randint(0, 2 ** 31)
EARLY_STOPPING = 3
lr = 0.005
reg = 0.01

def exp1():
    K = [[32], [64]]
    L = [2, 4, 8, 16]
    i = 1
    run_name = "exp1_1"
    for fpl in K:
        for lpb in L:
            print(f'Variation 1.{i}: {lpb=}, {fpl=}')
            i += 1
            run_experiment(run_name, filters_per_layer=fpl, layers_per_block=lpb, pool_every=4, reg=reg, lr=lr, out_dir=RESULTS_PATH, batches=BATCH_NUM, seed=SEED, early_stopping=EARLY_STOPPING)

def exp2():
    L = [2, 4, 8]
    K = [[32], [64], [128], [256]]
    i = 1
    run_name = "exp1_2"
    for lpb in L:
        for fpl in K:
            print(f'Variation 2.{i}: {lpb=}, {fpl=}')
            i += 1
            run_experiment(run_name, filters_per_layer=fpl, layers_per_block=lpb, pool_every=2, reg=reg, lr=lr, out_dir=RESULTS_PATH, batches=BATCH_NUM, seed=SEED, early_stopping=EARLY_STOPPING)

def exp3():
    K = [64, 128, 256]
    L = [1, 2, 3, 4]
    run_name = "exp1_3"
    i = 1
    for lpb in L:
        print(f'Variation 3.{i}: {lpb=}')
        i += 1
        run_experiment(run_name, filters_per_layer=K, layers_per_block=lpb, pool_every=6, reg=reg, lr=lr, out_dir=RESULTS_PATH, batches=BATCH_NUM, seed=SEED, early_stopping=EARLY_STOPPING)


def exp4():
    run_name = "exp1_4"
    K = [32]
    L = [8, 16, 32]
    i = 1

    for lpb in L:
        print(f'Variation 4.{i}: {lpb=}')
        i += 1
        run_experiment(run_name, model_type="resnet", filters_per_layer=K, layers_per_block=lpb, pool_every=8, lr=0.001, reg=reg, hidden_dims=[512], out_dir=RESULTS_PATH, batches=BATCH_NUM, seed=SEED, early_stopping=EARLY_STOPPING)
    
    K = [64, 128, 256]
    L = [2, 4, 8]

    for lpb in L:
        print(f'Variation 4.{i}: {lpb=}')
        i += 1
        run_experiment(run_name,  model_type="resnet", filters_per_layer=K, layers_per_block=lpb, pool_every=8, lr=0.001, reg=reg, hidden_dims=[512], out_dir=RESULTS_PATH, batches=BATCH_NUM, seed=SEED, early_stopping=EARLY_STOPPING)

def exp5():
    run_name = "exp2"
    K = [32, 64, 128]
    L = [3, 6, 9, 12]
    i = 1

    for lpb in L:
        print(f'Variation 5.{i}: {lpb=}')
        i += 1
        run_experiment(run_name, model_type="ycn", filters_per_layer=K, layers_per_block=lpb, pool_every=8, reg=0.03, lr=0.0007, hidden_dims=[512], out_dir=RESULTS_PATH, batches=BATCH_NUM, seed=SEED, early_stopping=EARLY_STOPPING)

if __name__ == "__main__":
    start = time()
    experiments = [exp1, exp2, exp3, exp4, exp5]
    args = argv[1:]
    print(f'{argv=} and {args=}')
    for e in args:
        e = int(e)
        experiments[e-1]()
    
    end = int(time() - start)
    m, s = end//60, end%60
    print(f'Total: {m}:{s} minutes')
    

