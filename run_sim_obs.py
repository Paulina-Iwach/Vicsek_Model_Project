import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

import numpy as np
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from numba import njit
import math

@njit
def line_circle_intersect_numba(x_old, x_new, cx, cy, R):
    ox = x_old[0]
    oy = x_old[1]
    nx = x_new[0]
    ny = x_new[1]
    dx = nx - ox
    dy = ny - oy
    fx = ox - cx
    fy = oy - cy
    A = dx*dx + dy*dy
    B = 2*(dx*fx + dy*fy)
    C = fx*fx + fy*fy - R*R

    if abs(A) < 1e-14:
        return -1.0, np.array([0.0, 0.0])
    
    disc = B*B - 4*A*C
    if disc < 0:
        return -1.0, np.array([0.0, 0.0])
    
    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)
    
    t_hit = 2.0
    if 0.0 <= t1 <= 1.0:
        t_hit = t1
    if 0.0 <= t2 <= 1.0 and t2 < t_hit:
        t_hit = t2
    if t_hit > 1.0:
        return -1.0, np.array([0.0, 0.0])
    
    p_hit = np.array([ox + t_hit*dx, oy + t_hit*dy])
    return t_hit, p_hit

@njit
def reflect_direction_numba(old_dir, nx, ny):
    """
    Odbicie wektora o kącie old_dir względem wektora normalnego (nx, ny).
    """
    vx = math.cos(old_dir)
    vy = math.sin(old_dir)
    norm_n = math.sqrt(nx*nx + ny*ny)
    if norm_n < 1e-14:
        return old_dir
    unx = nx / norm_n
    uny = ny / norm_n
    dot = vx*unx + vy*uny
    rx = vx - 2 * dot * unx
    ry = vy - 2 * dot * uny
    return math.atan2(ry, rx)

@njit
def process_collisions(positions, old_positions, new_dirs, obstacles, eps, L):
    """
    Przetwarza kolizje dla wszystkich cząsteczek, uwzględniając warunki periodyczne.
    
    Dla każdej cząsteczki:
      1. Obliczamy minimalny wektor przesunięcia między starą pozycją a nową (uwzględniając periodyczność),
         czyli „odwijamy” ruch agenta.
      2. Dla każdego obiektu przeszkody „odwijamy” także jego pozycję względem pozycji startowej agenta.
      3. Sprawdzamy przecięcia trajektorii agenta z przeszkodami i korygujemy ruch oraz kierunek.
      4. Na końcu „zapakowujemy” (modulo L) wynikową pozycję.
    """
    N = positions.shape[0]
    M = obstacles.shape[0]
    for i in range(N):
        # Obliczamy minimalny wektor przesunięcia (minimal image)
        dx = positions[i,0] - old_positions[i,0]
        dy = positions[i,1] - old_positions[i,1]
        if dx > L/2:
            dx -= L
        elif dx < -L/2:
            dx += L
        if dy > L/2:
            dy -= L
        elif dy < -L/2:
            dy += L
        x_old = old_positions[i].copy()  # Pozycja startowa
        x_new = np.empty(2)
        x_new[0] = x_old[0] + dx
        x_new[1] = x_old[1] + dy

        dir_old = new_dirs[i]
        min_t = 2.0
        best_obs_index = -1
        best_pt = np.zeros(2)
        
        # Szukamy najwcześniejszego przecięcia trajektorii agenta z przeszkodą
        for j in range(M):
            # "Odwijamy" środek przeszkody względem pozycji x_old
            cx = obstacles[j, 0]
            cy = obstacles[j, 1]
            dx_obs = cx - x_old[0]
            dy_obs = cy - x_old[1]
            if dx_obs > L/2:
                dx_obs -= L
            elif dx_obs < -L/2:
                dx_obs += L
            if dy_obs > L/2:
                dy_obs -= L
            elif dy_obs < -L/2:
                dy_obs += L
            cx_eff = x_old[0] + dx_obs
            cy_eff = x_old[1] + dy_obs
            
            R = obstacles[j, 2]
            t_hit, p_hit = line_circle_intersect_numba(x_old, x_new, cx_eff, cy_eff, R)
            if t_hit >= 0.0 and t_hit < min_t:
                min_t = t_hit
                best_obs_index = j
                best_pt = p_hit.copy()
                
        if min_t <= 1.0 and best_obs_index != -1:
            # Blok A: kolizja na torze ruchu
            # Ustalamy środek przeszkody dla best_obs_index (również "odwinięty")
            cx = obstacles[best_obs_index, 0]
            cy = obstacles[best_obs_index, 1]
            dx_obs = cx - x_old[0]
            dy_obs = cy - x_old[1]
            if dx_obs > L/2:
                dx_obs -= L
            elif dx_obs < -L/2:
                dx_obs += L
            if dy_obs > L/2:
                dy_obs -= L
            elif dy_obs < -L/2:
                dy_obs += L
            cx_eff = x_old[0] + dx_obs
            cy_eff = x_old[1] + dy_obs
            
            R = obstacles[best_obs_index, 2]
            nx = best_pt[0] - cx_eff
            ny = best_pt[1] - cy_eff
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-14:
                norm = 1.0
            unx = nx / norm
            uny = ny / norm
            # Odbicie – wyznaczamy nowy kąt
            new_angle = reflect_direction_numba(dir_old, unx, uny)
            # Weryfikacja: jeżeli nowy kąt nie kieruje wystarczająco na zewnątrz, wymuszamy kierunek wychodzący
            vx = math.cos(new_angle)
            vy = math.sin(new_angle)
            if vx*unx + vy*uny < 0.1:
                new_angle = math.atan2(uny, unx)
            new_dirs[i] = new_angle
            # Przesuwamy agenta poza przeszkodę
            x_new[0] = best_pt[0] + eps * unx
            x_new[1] = best_pt[1] + eps * uny
        else:
            # Blok B: nie wykryto przecięcia – sprawdzamy, czy końcowa pozycja (x_new) nie znajduje się wewnątrz przeszkody
            for j in range(M):
                cx = obstacles[j, 0]
                cy = obstacles[j, 1]
                dx_obs = cx - x_old[0]
                dy_obs = cy - x_old[1]
                if dx_obs > L/2:
                    dx_obs -= L
                elif dx_obs < -L/2:
                    dx_obs += L
                if dy_obs > L/2:
                    dy_obs -= L
                elif dy_obs < -L/2:
                    dy_obs += L
                cx_eff = x_old[0] + dx_obs
                cy_eff = x_old[1] + dy_obs
                
                R = obstacles[j, 2]
                dxc = x_new[0] - cx_eff
                dyc = x_new[1] - cy_eff
                dist = math.sqrt(dxc*dxc + dyc*dyc)
                if dist < R:
                    if dist < 1e-14:
                        unx = 1.0
                        uny = 0.0
                    else:
                        unx = dxc / dist
                        uny = dyc / dist
                    x_new[0] = cx_eff + (R + eps) * unx
                    x_new[1] = cy_eff + (R + eps) * uny
                    new_dirs[i] = math.atan2(uny, unx)
                    break
        # Dodatkowy pass – upewniamy się, że agent nie znajduje się wewnątrz żadnej przeszkody
        for j in range(M):
            cx = obstacles[j, 0]
            cy = obstacles[j, 1]
            dx_obs = cx - x_old[0]
            dy_obs = cy - x_old[1]
            if dx_obs > L/2:
                dx_obs -= L
            elif dx_obs < -L/2:
                dx_obs += L
            if dy_obs > L/2:
                dy_obs -= L
            elif dy_obs < -L/2:
                dy_obs += L
            cx_eff = x_old[0] + dx_obs
            cy_eff = x_old[1] + dy_obs
            
            R = obstacles[j, 2]
            dxc = x_new[0] - cx_eff
            dyc = x_new[1] - cy_eff
            dist = math.sqrt(dxc*dxc + dyc*dyc)
            if dist < R:
                if dist < 1e-14:
                    unx = 1.0
                    uny = 0.0
                else:
                    unx = dxc / dist
                    uny = dyc / dist
                x_new[0] = cx_eff + (R + eps) * unx
                x_new[1] = cy_eff + (R + eps) * uny
                new_dirs[i] = math.atan2(uny, unx)
                break
        
        # "Pakujemy" wynikową pozycję do przedziału [0, L)
        positions[i, 0] = x_new[0] % L
        positions[i, 1] = x_new[1] % L

#############################################
# Reszta symulacji (główna pętla)
#############################################

def vicsek_simulation_with_obstacle_reflect(
    N=300, L=10.0, r=1.0, eta=0.2, v=0.03,
    steps=500, dt=1.0, obstacles=None,
    store_trajectories=True
):
    """
    Model Vicseka z precyzyjnym odbiciem od przeszkód.
      - Szukanie sąsiadów przy użyciu cKDTree.
      - Obliczanie średniej orientacji + szum.
      - Aktualizacja pozycji z warunkami brzegowymi.
      - Sprawdzanie kolizji z przeszkodami – krytyczny fragment optymalizowany przez Numba.
    """
    eps = 1e-2

    if obstacles is None:
        obstacles = np.empty((0, 3), dtype=np.float64)
    else:
        obstacles = np.array(obstacles, dtype=np.float64)
    
    positions = np.random.uniform(0, L, (N, 2))
    directions = np.random.uniform(0, 2*np.pi, (N,))
    
    if store_trajectories:
        all_positions = np.empty((steps+1, N, 2), dtype=np.float32)
        all_directions = np.empty((steps+1, N), dtype=np.float32)
        all_positions[0] = positions.copy()
        all_directions[0] = directions.copy()
    else:
        all_positions, all_directions = None, None
    
    for step in range(steps):
        # 1) Znalezienie sąsiadów (używamy cKDTree)
        tree = cKDTree(positions, boxsize=L)
        neighbors_list = tree.query_ball_point(positions, r)
        for i, nbrs in enumerate(neighbors_list):
            if i not in nbrs:
                nbrs.append(i)
        
        # 2) Obliczamy średnią orientację + dodajemy szum
        sin_dir = np.sin(directions)
        cos_dir = np.cos(directions)
        new_dirs = np.empty_like(directions)
        for i, nbrs in enumerate(neighbors_list):
            avg_sin = np.mean(sin_dir[nbrs])
            avg_cos = np.mean(cos_dir[nbrs])
            noise = eta * (np.random.random() - 0.5)
            angle = np.arctan2(avg_sin, avg_cos) + noise
            new_dirs[i] = angle % (2*np.pi)
        
        # 3) Aktualizacja pozycji – najpierw zwykły ruch, potem warunki brzegowe
        old_positions = positions.copy()
        positions[:, 0] += v * np.cos(new_dirs) * dt
        positions[:, 1] += v * np.sin(new_dirs) * dt
        positions %= L  # warunki brzegowe
        
        # 4) Sprawdzamy kolizje – przekazujemy także L do funkcji, aby uwzględnić periodyczność
        process_collisions(positions, old_positions, new_dirs, obstacles, eps, L)
        
        # 5) Uaktualniamy kierunki
        directions = new_dirs.copy()
        
        # 6) Zapisujemy trajektorie, jeśli wymagane
        # if store_trajectories:
            # all_positions[step+1] = positions.copy()
        all_directions[step+1] = directions.copy()
    
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
    _, all_head = vicsek_simulation_with_obstacle_reflect(N=N, L=L, r=r, eta=eta, v=v, steps=steps, dt=dt,obstacles=[(L/2, L/2, L/5)], store_trajectories=store_trajectories)
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
        (400, 10),
        (100, 5),
        (40, 3.1),
        (4000, 31.6),
    ]
    
    eta_values = np.arange(0.0, 6.1, 0.1)  # Możesz zwiększyć liczbę punktów dla gęstszej siatki
    output_directory = "results_plot1_obs"
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
    output_directory = "results_plot2_rho_0_6_01_obs"
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
