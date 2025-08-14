import matplotlib.pyplot as plt
import numpy as np
from typing import List

# A placeholder for type hinting without needing the actual class import.
class GeneticActor:
    vector_return: np.ndarray

class BasicVisualizer:
    """
    A simple, no-frills visualizer to plot population performance.
    It uses one window and updates it in place.
    """
    def __init__(self, num_objectives: int):
        # Only plot if we have 2 or 3 objectives.
        if num_objectives not in [2, 3]:
            self.enabled = False
            return

        self.enabled = True
        self.num_objectives = num_objectives
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(8, 8))
        
        # Setup the plot for either 2D or 3D
        if num_objectives == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_zlabel("Objective 3")
        else:
            self.ax = self.fig.add_subplot(111)
        
        self.ax.set_xlabel("Objective 1")
        self.ax.set_ylabel("Objective 2")
        self.fig.show()

    def update(self, population: List[GeneticActor], generation: int):
        """
        The only function you need to call. It clears the plot and
        redraws the population's performance for the given generation.
        """
        if not self.enabled:
            return

        # 1. Clear the previous drawing
        self.ax.cla()

        # 2. Set titles and labels for the new drawing
        self.ax.set_title(f"Population Performance: Generation {generation}")
        self.ax.set_xlabel("Objective 1")
        self.ax.set_ylabel("Objective 2")
        if self.num_objectives == 3:
            self.ax.set_zlabel("Objective 3")

        # 3. Get all valid performance scores from the population
        points = [p.vector_return for p in population if p.vector_return is not None]
        if not points:
            return # Don't plot if there's nothing to show

        returns_matrix = np.array(points)

        # 4. Create the scatter plot
        if self.num_objectives == 3:
            self.ax.scatter(returns_matrix[:, 0], returns_matrix[:, 1], returns_matrix[:, 2], c='blue', alpha=0.8)
        else:
            self.ax.scatter(returns_matrix[:, 0], returns_matrix[:, 1], c='blue', alpha=0.8)

        self.ax.grid(True, linestyle='--', alpha=0.5)

        # 5. Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()