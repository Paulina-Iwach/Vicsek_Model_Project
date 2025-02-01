import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

import numpy as np
from scipy.spatial import cKDTree

def line_circle_intersect(x_old, x_new, cx, cy, R):
    """
    Znajduje przecięcie odcinka x_old -> x_new z okręgiem (cx, cy, R).
    x_old: (2,) stara pozycja
    x_new: (2,) nowa pozycja
    Zwraca (t, point), gdzie t w [0,1], point = (px, py) jest punktem przecięcia
    jeśli istnieje. Jeśli brak przecięcia lub leży poza odcinkiem [0,1], 
    zwraca (None, None).
    
    Rozwiązujemy || x_old + t*(x_new - x_old) - c || = R
    tj. || (dx, dy)*t + (x_old - c) || = R.
    """
    ox, oy = x_old
    nx, ny = x_new
    dx = nx - ox
    dy = ny - oy
    
    # Wektor do środka okręgu
    fx = ox - cx
    fy = oy - cy
    
    # Parametry równania kwadratowego
    # (dx^2 + dy^2)*t^2 + 2(dx fx + dy fy)*t + (fx^2 + fy^2 - R^2) = 0
    A = dx*dx + dy*dy
    B = 2*(dx*fx + dy*fy)
    C = fx*fx + fy*fy - R*R
    
    # Jeśli A=0 (x_old==x_new?), raczej brak ruchu
    if abs(A) < 1e-14:
        return None, None
    
    disc = B*B - 4*A*C
    if disc < 0:
        # brak rzeczywistego przecięcia
        return None, None
    
    # Mamy 2 rozwiązania t1, t2:
    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)
    
    # Szukamy "najmniejszego dodatniego" t, bo to pierwsze przecięcie z odcinkiem
    sol = []
    for tt in [t1, t2]:
        if 0 <= tt <= 1:
            sol.append(tt)
    
    if len(sol)==0:
        return None, None
    
    t_hit = min(sol)
    px = ox + t_hit*dx
    py = oy + t_hit*dy
    return t_hit, (px, py)

def reflect_direction(old_dir, normal_vec):
    """
    Odbicie wektora z kąta old_dir (radian) względem normal_vec (2,).
    normal_vec nie musi być znormalizowany, ważny jest kierunek.
    Zwraca new_dir (radian).
    
    Algorytm:
      v = (cos old_dir, sin old_dir)
      n = normal_vec (unorm)
      1) un = n / ||n||
      2) v' = v - 2(v dot un)*un
      3) new_dir = atan2(v'_y, v'_x)
    """
    vx = np.cos(old_dir)
    vy = np.sin(old_dir)
    nx, ny = normal_vec
    
    norm_n = np.sqrt(nx*nx + ny*ny)
    if norm_n < 1e-14:
        return old_dir  # normal degenerate => no reflection
    
    unx = nx / norm_n
    uny = ny / norm_n
    
    dot = vx*unx + vy*uny
    rx = vx - 2*dot*unx
    ry = vy - 2*dot*uny
    
    return np.arctan2(ry, rx)

def inside_any_obstacle(pos, obstacles):
    """
    Czy pos=(x,y) leży w środku którejś z przeszkód?
    """
    x, y = pos
    for cx, cy, R in obstacles:
        dx = x - cx
        dy = y - cy
        if dx*dx + dy*dy < R*R:
            return True
    return False

def vicsek_simulation_with_obstacle_reflect(
    N=300, L=10.0, r=1.0, eta=0.2, v=0.03,
    steps=500, dt=1.0, obstacles=None,
    store_trajectories=True
):
    """
    Modyfikacja modelu Vicseka:
     - cKDTree do sąsiadów,
     - Okolice przeszkód: jeśli tor od x_old do x_new przecina okrąg,
       to cząstka kończy ruch w punkcie przecięcia i odbija kierunek
       względem normalnej do okręgu.
     - Jeśli tor nie przecina, ale finał wypadł "wewnątrz" (skrajne),
       to stawiamy na brzegu w najbliższym punkcie i odbijamy.
    """
    if obstacles is None:
        obstacles = []
    
    positions = np.random.uniform(0, L, (N, 2))
    directions = np.random.uniform(0, 2*np.pi, (N,))
    
    if store_trajectories:
        all_positions = np.zeros((steps+1, N, 2))
        all_directions = np.zeros((steps+1, N))
        all_positions[0] = positions
        all_directions[0] = directions
    else:
        all_positions, all_directions = None, None
    
    for step in range(steps):
        # 1) cKDTree
        tree = cKDTree(positions, boxsize=L)
        neighbors_list = tree.query_ball_point(positions, r, n_jobs=-1)
        
        # Self-inclusion (Vicsek standard)
        for i, nbrs in enumerate(neighbors_list):
            if i not in nbrs:
                nbrs.append(i)
        
        # 2) Średnie kąty + szum
        sin_dir = np.sin(directions)
        cos_dir = np.cos(directions)
        new_dirs = np.zeros_like(directions)
        
        for i, nbrs in enumerate(neighbors_list):
            sum_sin = np.sum(sin_dir[nbrs])
            sum_cos = np.sum(cos_dir[nbrs])
            ccount = len(nbrs)
            avg_sin = sum_sin / ccount
            avg_cos = sum_cos / ccount
            
            noise = eta*(np.random.random()-0.5)  # [-eta/2, +eta/2]
            angle = np.arctan2(avg_sin, avg_cos) + noise
            new_dirs[i] = angle % (2*np.pi)
        
        # 3) Aktualizujemy pozycje kandydujące
        old_positions = positions.copy()
        positions[:,0] += v*np.cos(new_dirs)*dt
        positions[:,1] += v*np.sin(new_dirs)*dt
        # Periodyczne brzegowe
        positions %= L
        
        # 4) Sprawdzamy kolizje z przeszkodami 
        #    (szukamy najmniejszego t, jesli kilka okręgów)
        for i in range(N):
            x_old = old_positions[i]
            x_new = positions[i]
            dir_old = new_dirs[i]
            
            # Krok A: sprawdzamy wszystkie przeszkody, 
            #         czy jest przecięcie w [0,1]?
            min_t = 2.0
            best_pt = None
            best_obs = None
            for (cx, cy, R) in obstacles:
                t_hit, p_hit = line_circle_intersect(x_old, x_new, cx, cy, R)
                if t_hit is not None and t_hit<min_t:
                    min_t = t_hit
                    best_pt = p_hit
                    best_obs = (cx, cy, R)
            
            # Jeśli min_t <=1 => mamy kolizję "w trakcie" ruchu
            if min_t <=1.0 and best_pt is not None:
                # Wstawiamy w punkt kolizji
                positions[i] = best_pt
                # Odbijamy kierunek
                cx, cy, R = best_obs
                nx = best_pt[0] - cx
                ny = best_pt[1] - cy
                new_dirs[i] = reflect_direction(dir_old, (nx, ny))
                
            else:
                # Krok B: sprawdź, czy finał nie jest jednak w środku
                if inside_any_obstacle(positions[i], obstacles):
                    # Wtedy stawiamy na brzegu w najbliższym punkcie 
                    # i odbijamy w stronę normalną
                    # (co oznacza, że wniknęliśmy - 
                    #  to nieco bardziej "agresywne" wniknięcie, 
                    #  więc może się rzadko zdarzać)
                    # 
                    # Znajdujemy obstacle, do którego wpadł 
                    # (tu upraszczamy, że jest tylko jeden).
                    for (cx, cy, R) in obstacles:
                        dx = positions[i,0]-cx
                        dy = positions[i,1]-cy
                        dist = np.hypot(dx,dy)
                        if dist < R:
                            # stawiamy na brzegu 
                            ratio = R/dist
                            bx = cx + dx*ratio
                            by = cy + dy*ratio
                            positions[i,0]=bx
                            positions[i,1]=by
                            # normalna 
                            new_dirs[i] = reflect_direction(
                                new_dirs[i], (dx,dy)
                            )
                            break
        
        # 5) Zatwierdzamy kierunki
        directions = new_dirs
        
        # 6) Zapis
        if store_trajectories:
            all_positions[step+1] = positions
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
