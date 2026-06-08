import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from typing import Sequence

class RectangleDef:
    """Definition of rectangle"""
    def __init__(self, id, w, h, v):
        self.id = id
        self.w = w
        self.h = h
        self.v = v

class PlacedRectangle:
    """Representation of fitted rectangle"""
    def __init__(self, rect_def, x, y):
        self.rect = rect_def
        self.x = x
        self.y = y

    def intersects(self, other):
        """Checks if two rectangles are intersecting one another"""
        return not (self.x + self.rect.w <= other.x or 
                    other.x + other.rect.w <= self.x or 
                    self.y + self.rect.h <= other.y or 
                    other.y + other.rect.h <= self.y)


class CuttingStockEA:
    def __init__(self, radius, rect_types, population_size=50):
        self.r = radius
        self.rect_types = rect_types

        def nwd(a, b):
            if a == b:
                return a
            if b > a:
                a = a + b
                b = a - b
                a = a - b
            return nwd(a-b, b)

        def nwd_lista(lista):
            nwd_val = nwd(lista[0], lista[1])
            for i in range(2, len(lista)):
                nwd_val = nwd(nwd_val, lista[i])
            return nwd_val

        dim_list = [[rect.w, rect.h] for rect in rect_types]
        dim_list = np.array(dim_list).flatten()

        self.step = nwd_lista(dim_list)
        # nwd to choose as long step as possible to optimise runtime
        # but still short enought to perfectly fit rectangles

        self.population_size = population_size
        # Estimating max chromosom length (circle area / area of smallest rectangle)
        min_area = min(r.w * r.h for r in rect_types)
        circle_area = math.pi * (self.r ** 2)
        self.chrom_length = int(circle_area / min_area)
        
        # Parametry ewolucji
        self.p_crossover = 0.8
        self.p_mutation = 0.2
        self.tournament_size = 5


        self.population = [np.random.randint(0, len(self.rect_types), self.chrom_length).tolist() 
                      for _ in range(self.population_size)]
        self.fitness_scores = np.zeros(self.population_size)


    def _fits_in_circle(self, x, y, w, h):
        """Sprawdza, czy wszystkie 4 wierzchołki leżą wewnątrz koła (0,0)."""
        corners = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
        for cx, cy in corners:
            if cx**2 + cy**2 > self.r**2:
                return False
        return True

    def _decode_and_evaluate(self, chromosome):
        
        placed: list = []
        placed_fast: list = [] # (x, y, w, h)
        total_value: float = 0.0
        
        step = self.step
        r_sq = self.r ** 2 

        failed_rect_types: set = set()

        num_unique_rect_types: int = len(set(chromosome))

        for rect_idx in chromosome:
            if len(failed_rect_types) == num_unique_rect_types:
                # we already check all rect_types in chromosome
                # no need to the rest of them
                break
            if rect_idx in failed_rect_types: 
                # dont try to place rect type that we failed already
                continue

            rect = self.rect_types[rect_idx]
            rect_w = rect.w
            rect_h = rect.h
            placed_successfully = False
            
            y = -self.r
            while y <= self.r - rect_h:   
                y_top = y + rect_h

                abs_y  = -y     if y     < 0 else y
                abs_yt = -y_top if y_top < 0 else y_top
                max_y_sq = abs_y * abs_y if abs_y > abs_yt else abs_yt * abs_yt
                
                if max_y_sq > r_sq:
                    y += step
                    continue
                
                # The inner x-loop can then skip the y-overlap test entirely.
                active = [(px, py, pw, ph) for px, py, pw, ph in placed_fast
                          if y_top > py and py + ph > y]

                max_x_abs = math.sqrt(r_sq - max_y_sq)
                x = -max_x_abs
                x_end = max_x_abs - rect_w
                
                while x <= x_end:
                    x_right = x + rect_w
                    next_x = x + step
                    collision = False

                    for px, py, pw, ph in active:
                        if x_right > px and px + pw > x:   # y already filtered above
                            collision = True
                            # (5) Inline max: jump past this blocker's right edge
                            if px + pw > next_x:
                                next_x = px + pw
                            break   # break-on-first preserves original placement order
                    
                    if not collision:
                        placed.append(PlacedRectangle(rect, x, y))
                        placed_fast.append((x, y, rect_w, rect_h)) 
                        total_value += rect.v
                        placed_successfully = True
                        break 
                        
                    x = next_x
                if placed_successfully:
                    break
                y += step
            
            if not placed_successfully:
                failed_rect_types.add(rect_idx)

        return total_value, placed

    def _tournament_selection(self, t_size=None, minimize=False):
        """Returns index of the best individual in the tournament"""
        if t_size is None:
            t_size = self.tournament_size
        indices = np.random.choice(self.population_size, t_size, replace=False)
        if minimize:
            best_index = indices[np.argmin(self.fitness_scores[indices])]
        else:
            best_index = indices[np.argmax(self.fitness_scores[indices])]
        return best_index

    def _weighted_crossover(self, parent1, parent2, fitness1, fitness2):
        """Better parent has higher probability of getting choosen."""
        suma_ocen = fitness1 + fitness2
        p_A = fitness1 / suma_ocen if suma_ocen > 0 else 0.5
        
        np_p1 = np.array(parent1)
        np_p2 = np.array(parent2)
        
        maska = np.random.rand(self.chrom_length) < p_A
        child_np = np.where(maska, np_p1, np_p2)
        return child_np.tolist()

    def _weighted_crossover_varying_probability(self, parent1, parent2, fitness1, fitness2):
        """Rectangles at the back are more likely to be changed."""
        suma_ocen = fitness1 + fitness2
        p_A = fitness1 / suma_ocen if suma_ocen > 0 else 0.5
        
        np_p1 = np.array(parent1)
        np_p2 = np.array(parent2)
        
        maska = np.random.rand(self.chrom_length) < p_A
        ch_l = self.chrom_length
        for i in range(ch_l):
            if np.random.rand() < 0.5 + (i+1)/2*ch_l:
                continue
            if np.random.rand() < p_A:
                maska[i] = True
            else:
                maska[i] = False

        child_np = np.where(maska, np_p1, np_p2)
        return child_np.tolist()


    def run(self, generations=100):
        num_types = len(self.rect_types)
        best_overall_score = 0
        best_overall_placement = []
        top_tournament_size = max(self.population_size // 4, 3)
        for gen in range(generations):
            current_placements = []
            print(f"Generacja: {gen}")
            for i, chrom in enumerate(self.population):
                score, placement = self._decode_and_evaluate(chrom)
                self.fitness_scores[i] = score
                current_placements.append(placement)
            
            scores_np = np.array(self.fitness_scores)
            sorted_indices = np.argsort(scores_np)[::-1]
            current_best_idx = sorted_indices[0]
            
            if self.fitness_scores[current_best_idx] > best_overall_score:
                best_overall_score = self.fitness_scores[current_best_idx]
                
                best_overall_placement = current_placements[current_best_idx] 
                print(f"Generacja {gen}: Nowy najlepszy wynik = {best_overall_score}")

            new_population = []
            
            n_best = 3
            for i in range(n_best):
                idx = sorted_indices[i]
                new_population.append(list(self.population[idx].copy()))
            
            while len(new_population) < self.population_size:
                p1_id = self._tournament_selection(t_size=top_tournament_size) # one parent is usually very good
                p2_id = self._tournament_selection()

                f1 = self.fitness_scores[p1_id]
                f2 = self.fitness_scores[p2_id]
                
                # child = self._weighted_crossover_varying_probability(
                #     self.population[p1_id], self.population[p2_id], f1, f2)
                
                child = self._weighted_crossover(
                    self.population[p1_id], self.population[p2_id], f1, f2)
                
                # if random.random() < self.p_crossover:
                # else:
                #     child = list(self.population[p1_id].copy())

                for i in range(self.chrom_length):
                    if random.random() < self.p_mutation:
                        child[i] = random.randint(0, num_types - 1)

                new_population.append(child)

            self.population = new_population

        return best_overall_score, best_overall_placement
    

    def plot_solution(self, placed_rectangles):
        """
        Rysuje koło oraz wstawione w nim prostokąty.
        Różne typy prostokątów (ID) otrzymują różne kolory.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        circle = plt.Circle((0, 0), self.r, color='lightgray', fill=True, 
                            linestyle='--', edgecolor='black', linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)
        
        cmap = plt.get_cmap('tab20')
        
        for p in placed_rectangles:
            color = cmap(p.rect.id % 20)
            
            rect_patch = patches.Rectangle(
                (p.x, p.y), p.rect.w, p.rect.h,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.9
            )
            ax.add_patch(rect_patch)
            
            center_x = p.x + p.rect.w / 2
            center_y = p.y + p.rect.h / 2
            
            ax.text(center_x, center_y, str(p.rect.v), color='white', 
                    weight='bold', fontsize=8, ha='center', va='center',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        limit = self.r * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_title(f"Rozkrój koła (Promień: {self.r})\nŁączna wartość: {sum(p.rect.v for p in placed_rectangles)}", fontsize=14)
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()