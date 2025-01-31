import cupy as cp
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def vicsek_simulation_gpu(
    N=300, L=7.0, r=1.0, eta=0.2, v=0.03, 
    steps=500, dt=1.0, store_trajectories=True
):
    """
    A GPU-accelerated version of the Vicsek model simulation using CuPy.
    """
    positions = L * cp.random.rand(N, 2)
    directions = 2 * cp.pi * cp.random.rand(N)

    if store_trajectories:
        all_positions = [positions.copy()]
        all_directions = [directions.copy()]
    else:
        all_positions = []
        all_directions = []

    r_sq = r * r

    for step in range(steps):
        dx = positions[:, cp.newaxis, 0] - positions[:, 0]
        dy = positions[:, cp.newaxis, 1] - positions[:, 1]

        dx -= cp.round(dx / L) * L
        dy -= cp.round(dy / L) * L

        dist_sq = dx**2 + dy**2
        neighbors = dist_sq < r_sq

        cp.fill_diagonal(neighbors, False)

        sin_dir = cp.sin(directions)
        cos_dir = cp.cos(directions)

        sum_sin = neighbors @ sin_dir
        sum_cos = neighbors @ cos_dir

        count_neighbors = neighbors.sum(axis=1)
        count_neighbors = cp.where(count_neighbors == 0, 1, count_neighbors)

        avg_sin = sum_sin / count_neighbors
        avg_cos = sum_cos / count_neighbors

        noise = eta * (cp.random.rand(N) - 0.5)
        new_directions = cp.arctan2(avg_sin, avg_cos) + noise
        new_directions = new_directions % (2 * cp.pi)

        positions[:, 0] += v * cp.cos(new_directions) * dt
        positions[:, 1] += v * cp.sin(new_directions) * dt

        positions %= L
        directions = new_directions

        if store_trajectories:
            all_positions.append(positions.copy())
            all_directions.append(directions.copy())

    if store_trajectories:
        all_positions = [cp.asnumpy(pos) for pos in all_positions]
        all_directions = [cp.asnumpy(dir) for dir in all_directions]
    else:
        all_positions = []
        all_directions = [cp.asnumpy(dir) for dir in all_directions]

    return all_positions, all_directions

def run_simulation_and_save_gpu(N, L, eta, r=1.0, v=0.03, steps=100, dt=1.0, store_trajectories=True, output_dir="results_gpu"):
    """
    Runs the GPU-accelerated Vicsek simulation and saves the data to a compressed .npz file.
    """
    all_positions, all_head = vicsek_simulation_gpu(N=N, L=L, r=r, eta=eta, v=v, steps=steps, dt=dt, store_trajectories=store_trajectories)
    
    os.makedirs(output_dir, exist_ok=True)
    
    eta_str = f"{eta:.2f}".replace('.', '_')
    filename = f"simulation_gpu_N{N}_L{L}_eta{eta_str}.npz"
    filepath = os.path.join(output_dir, filename)
    
    if store_trajectories:
        all_positions_array = np.array(all_positions)      # Shape: (steps+1, N, 2)
        all_head_array = np.array(all_head)              # Shape: (steps+1, N)
        np.savez_compressed(filepath, all_positions=all_positions_array, all_directions=all_head_array)
    else:
        all_head_array = np.array(all_head)
        np.savez_compressed(filepath, all_directions=all_head_array)
    
    return eta

def main():
    systems = [
        #(10000, 50),
        #(4000, 31.6),
        #(400, 10),
        (100, 5),
        (40, 3.1)
    ]
    eta_values = np.arange(0.0, 5.0, 0.05)
    output_directory = "results_gpu"

    for (N, L) in systems:
        print(f"Rozpoczynanie symulacji na GPU dla N={N}, L={L}")
        
        Parallel(n_jobs=-1)(
            delayed(run_simulation_and_save_gpu)(
                N=N, 
                L=L, 
                eta=eta, 
                r=1.0, 
                v=0.03, 
                steps=10**4, 
                dt=1.0, 
                store_trajectories=True, 
                output_dir=output_directory
            ) 
            for eta in tqdm(eta_values, desc=f"Symulacje na GPU dla N={N}, L={L}")
        )
        
        print(f"Symulacje na GPU dla N={N}, L={L} zakończone. Wyniki zapisane w katalogu '{output_directory}'.\n")

if __name__ == "__main__":
    main()
