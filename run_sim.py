import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

import numpy as np
from scipy.spatial import cKDTree

def vicsek_simulation_ckdtree_vectorized(
    N=300, L=7.0, r=1.0, eta=0.2, v=0.03, 
    steps=500, dt=1.0, store_trajectories=True
):
    """
    Zwektoryzowana implementacja modelu Vicseka z użyciem cKDTree.
    - Pozycje i kierunki inicjalizowane losowo.
    - W każdym kroku:
       1) Tworzymy cKDTree(positions, boxsize=L).
       2) Dla każdej cząstki szukamy sąsiadów w promieniu r -> neighbors_list
       3) Tworzymy 2D neighbor_array (N, max_neighbors) z -1 na wolnych miejscach
       4) Sumujemy sin(y) i cos(y) wektorowo, obliczamy średni kąt + szum
       5) Aktualizujemy pozycje (periodycznie) i zapisujemy (opcjonalnie) trajektorie.
    Zwraca (all_positions, all_directions) lub (None, None) w zależności od 
    store_trajectories.
    """

    # --- Inicjalizacja ---
    positions = np.random.uniform(0, L, (N, 2))     # (N,2) położenia
    directions = np.random.uniform(0, 2*np.pi, N)   # (N,) kierunki w [0,2π)
    
    if store_trajectories:
        # Zapis w tablicach (steps+1) x N x 2 i (steps+1) x N
        # albo w listach - tutaj tablice:
        all_positions = np.empty((steps+1, N, 2), dtype=np.float32)
        all_directions = np.empty((steps+1, N), dtype=np.float32)
        all_positions[0] = positions
        all_directions[0] = directions
    else:
        all_positions = None
        all_directions = None
    
    # --- Główna pętla czasowa ---
    for step in range(steps):
        # 1) Budowa cKDTree (uwzględnia warunki periodyczne przez boxsize=L)
        tree = cKDTree(positions, boxsize=L)
        
        # 2) Lista list sąsiadów
        #    query_ball_point zwraca listę z N elementami; element i to lista sąsiadów i-tej cząstki
        neighbors_list = tree.query_ball_point(positions, r)
        
        # 2a) Opcjonalnie włączamy cząstkę samą w sobie (tak jak w oryginalnym Vicseku)
        for i, nbrs in enumerate(neighbors_list):
            if i not in nbrs:
                nbrs.append(i)
        
        # 2b) Określamy, ile maksymalnie sąsiadów występuje
        max_neighbors = max(len(nbrs) for nbrs in neighbors_list)
        
        # 2c) Tworzymy 2D tablicę o wymiarze (N, max_neighbors), wypełnioną -1
        neighbor_array = np.full((N, max_neighbors), -1, dtype=np.int32)
        
        # 2d) Wypełniamy neighbor_array indeksami sąsiadów
        for i, nbrs in enumerate(neighbors_list):
            neighbor_array[i, :len(nbrs)] = nbrs
        
        # 3) Wektorowo liczymy sumy sin i cos
        sin_dir = np.sin(directions)
        cos_dir = np.cos(directions)
        
        # Maska True/False = czy dany "slot" w neighbor_array jest obsadzony
        valid_mask = (neighbor_array != -1)  # shape (N, max_neighbors)
        
        # sin_dir[neighbor_array] - to shape (N, max_neighbors), ale -1 jest niepoprawnym indeksem.
        # Dlatego mnożymy przez valid_mask, by wyzerować wkład z -1.
        sum_sin = np.sum( sin_dir[neighbor_array] * valid_mask, axis=1 )
        sum_cos = np.sum( cos_dir[neighbor_array] * valid_mask, axis=1 )
        
        # Liczba sąsiadów
        count_neighbors = valid_mask.sum(axis=1)
        
        # Unikamy dzielenia przez zero (w modelu Vicseka z self in neighbors i tak count>=1)
        # Niemniej warto się zabezpieczyć:
        count_neighbors = np.where(count_neighbors==0, 1, count_neighbors)
        
        # Średnie sin i cos
        avg_sin = sum_sin / count_neighbors
        avg_cos = sum_cos / count_neighbors
        
        # Nowy kąt = arctan2(...) + noise
        # Noise w [-eta/2, +eta/2]
        noise = eta * (np.random.rand(N) - 0.5)
        
        new_directions = np.arctan2(avg_sin, avg_cos) + noise
        new_directions %= (2*np.pi)  # zawijamy w [0,2π)
        
        # 4) Aktualizacja położeń (wektorowo)
        positions[:,0] += v * np.cos(new_directions) * dt
        positions[:,1] += v * np.sin(new_directions) * dt
        positions %= L  # periodyczne warunki brzegowe
        
        # 5) Przepisujemy directions
        directions = new_directions
        
        # 6) Zapis do trajektorii, jeśli potrzebne
        # if store_trajectories:
        # all_positions[step+1] = positions
        all_directions[step+1] = directions
    
    return all_positions, all_directions


def measure_order_parameter_vectorized(all_directions):
    # Obliczanie cos(i kierunku) i sin(i kierunku) dla wszystkich cząstek i kroków czasowych
    cos_dirs = np.cos(all_directions)  # Kształt: (steps, N)
    sin_dirs = np.sin(all_directions)  # Kształt: (steps, N)
    
    # Sumowanie po osi N (cząstki) dla każdego kroku czasowego
    sum_cos = np.sum(cos_dirs, axis=1)  # Kształt: (steps,)
    sum_sin = np.sum(sin_dirs, axis=1)  # Kształt: (steps,)
    
    # Obliczanie normy sum wektorowych
    sum_norm = np.sqrt(sum_cos**2 + sum_sin**2)  # Kształt: (steps,)
    
    # Obliczanie parametru porządku v_a
    v_a = sum_norm / all_directions.shape[1]  # Zakładamy, że all_directions.shape[1] = N
    
    return v_a


def run_simulation_and_save(N, L, eta, r=1.0, v=0.03, steps=100, dt=1.0, store_trajectories=True, output_dir="results"):
    # Run the simulation
    _, all_head = vicsek_simulation_ckdtree_vectorized(N=N, L=L, r=r, eta=eta, v=v, steps=steps, dt=dt, store_trajectories=store_trajectories)
    all_directions_array = np.array(all_head)  # Kształt: (steps, N)
    va_array = measure_order_parameter_vectorized(all_directions_array)
    va_mean = np.mean(va_array)
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename-friendly eta string
    eta_str = f"{eta:.2f}".replace('.', '_')
    filename = f"va_mean_N{N}_L{L}_eta{eta_str}.npy"  # Zmiana rozszerzenia na .npy
    filepath = os.path.join(output_dir, filename)
    
    # Convert all_head (list of ndarrays) to a 2D array (steps+1, N)
    all_head_array = np.array(va_mean)  # Shape: (steps+1, N)
    
    # Save to a .npy file
    np.save(filepath, all_head_array)
    
    return eta


def main():
    # Definiowanie systemów i wartości eta
    systems = [
        (4000, 31.6),
        (400, 10),
        (100, 5),
        (40, 3.1)
    ]
    
    eta_values = np.arange(0.0, 6.1, 0.1)  # Możesz zwiększyć liczbę punktów dla gęstszej siatki
    output_directory = "results_plot1"
    for (N, L) in systems:
        print(f"Rozpoczynanie symulacji dla N={N}, L={L}")
        
        # Użycie Parallel i delayed do zrównoleglenia symulacji
        # Wrap the eta_values with tqdm for a progress bar
        # Funkcja run_simulation_and_save zostanie uruchomiona równolegle dla każdej wartości eta
        Parallel(n_jobs=-1)(
            delayed(run_simulation_and_save)(
                N=N, 
                L=L, 
                eta=eta, 
                r=1.0, 
                v=0.03, 
                steps=2*10**4, 
                dt=1.0, 
                store_trajectories=True, 
                output_dir=output_directory
            ) 
            for eta in tqdm(eta_values, desc=f"Symulacje dla N={N}, L={L}")
        )
        print(f"Symulacje dla N={N}, L={L} zakończone. Wyniki zapisane w katalogu '{output_directory}'.\n")


    rhos = np.arange(0.1, 6.1, 0.1)  # Możesz zwiększyć liczbę punktów dla gęstszej siatki
    L_fixed = 20
    Ns = rhos*L_fixed**2
    output_directory = "results_plot2_rho_0_6_01"
    Parallel(n_jobs=-1)(
        delayed(run_simulation_and_save)(
            N=int(np.round(N)), 
            L=20, 
            eta=2., 
            r=1.0, 
            v=0.03, 
            steps=2*10**4, 
            dt=1.0, 
            store_trajectories=True, 
            output_dir=output_directory
        ) 
        for N in tqdm(Ns, desc=f"Symulacje2")
    )



if __name__ == "__main__":
    main()
