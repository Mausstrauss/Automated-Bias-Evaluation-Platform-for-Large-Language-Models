import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

class BiasVisualizer:
    """
    Generates the 'Multi-Axis Bias Profile' (Radar Chart).
    This is the 'Artist' that draws the visualization for both the CLI and Web App.
    """
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_radar_profile(self, profile_data: dict, model_name: str):
        """
        Creates a Radar Chart where each axis is a bias category.
        """
        if not profile_data:
            return

        # 1. Prepare Data
        categories = list(profile_data.keys())
        values = list(profile_data.values())
        N = len(categories)

        # We need to repeat the first value to close the circle
        values += values[:1]
        
        # Calculate angles for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # 2. Setup Plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable + labels
        plt.xticks(angles[:-1], categories, color='grey', size=10)
        
        # Draw y-labels (0.2, 0.4, ... 1.0)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                   color="grey", size=7)
        plt.ylim(0, 1.0)

        # 3. Plot Data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name)
        ax.fill(angles, values, 'b', alpha=0.1)

        # 4. Finish
        plt.title(f"Bias Profile: {model_name}", size=15, y=1.1)
        
        # Save to file (Mainly for the CLI version)
        filename = f"{self.output_dir}/bias_radar_{model_name.replace('/', '_')}.png"
        plt.savefig(filename)
        
        # Close plot to free memory (Important for web apps!)
        plt.close(fig)