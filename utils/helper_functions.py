import numpy as np
import pandas as pd
import os

def create_synthetic_mobile_data(filename):
    print("Generating synthetic mobile dataset...")
    np.random.seed(42)
    n = 300
    ram = np.random.choice([8, 12, 16, 24], n)
    storage = np.random.choice([128, 256, 512, 1024, 2048], n)
    battery = np.random.choice([4000, 4500, 5000, 5500, 6000], n)
    camera = np.random.choice([48, 50, 64, 108, 200], n)
    refresh_rate = np.random.choice([60, 90, 120, 144, 165], n)
    
    price = (ram * 1200) + (storage * 35) + (battery * 2) + (camera * 120) + (refresh_rate * 50) + 20000
    flagship_mask = (ram >= 12) & (storage >= 512)
    price[flagship_mask] += 15000
    price += np.random.randint(-4000, 4000, n)
    
    df = pd.DataFrame({'RAM': ram, 'Storage': storage, 'Battery': battery, 'Camera': camera, 'RefreshRate': refresh_rate, 'Price': price})
    df.to_csv(filename, index=False)

def create_synthetic_laptop_data(filename):
    print("Generating synthetic laptop dataset...")
    np.random.seed(42)
    n = 300
    ram = np.random.choice([8, 16, 32, 64, 128], n)
    storage = np.random.choice([256, 512, 1024, 2048, 4096], n)
    gpu = np.random.choice(['Integrated', 'RTX 5060', 'RTX 5080', 'M5 Max', 'M5 Pro'], n)
    screen = np.random.choice([13.3, 14.0, 15.6, 16.0, 17.3, 18.0], n)
    processor = np.random.choice(['Base', 'Pro', 'Max', 'Ultra'], n)
    
    gpu_cost = {'Integrated': 0, 'RTX 5060': 25000, 'RTX 5080': 65000, 'M5 Pro': 50000, 'M5 Max': 110000}
    proc_cost = {'Base': 10000, 'Pro': 25000, 'Max': 45000, 'Ultra': 75000}
    
    price = np.zeros(n)
    for i in range(n):
        price[i] = (ram[i] * 900) + (storage[i] * 32) + (screen[i] * 600) + gpu_cost[gpu[i]] + proc_cost[processor[i]] + 35000
        price[i] += np.random.randint(-8000, 8000)
        
    df = pd.DataFrame({'RAM': ram, 'Storage': storage, 'GPU': gpu, 'Screen': screen, 'Processor': processor, 'Price': price})
    df.to_csv(filename, index=False)

def ensure_datasets_exist(mobile_csv, laptop_csv):
    if not os.path.exists(mobile_csv):
        create_synthetic_mobile_data(mobile_csv)
    if not os.path.exists(laptop_csv):
        create_synthetic_laptop_data(laptop_csv)
