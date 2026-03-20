import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory data
data = np.load('example_datasets/processed/arctic/leap_hand/right/scissors-25-186-s02-u01/0/trajectory_dexmachina_4.0_4.0_10.0_13.0.npz')

# --- EASY CONFIGURATION ---
keys_to_visualize = [
    'rew_mean',
    'obj_dist_rew_mean',
    'obj_arti_rew_mean',
    'contact_rew_mean',
    'obj_pos_dist_mean',
    'obj_quat_dist_mean',
    'imi_rew_mean',
    'improvement',
    'fingertip_dist_right_mean',
]

def plot_all_metrics(data, keys):
    # 1. Identify valid keys and determine the number of stages dynamically
    valid_keys = [k for k in keys if k in data]
    if not valid_keys:
        print("No valid keys found in data.")
        return

    # Dynamically get the number of stages from the first valid array found
    num_stages = data[valid_keys[0]].shape[0]

    # Generate labels automatically (e.g., "Stage 1", "Stage 2"...)
    # Or keep your fraction style if you prefer: [f"{i+1}/{num_stages}" for i in range(num_stages)]
    opt_labels = [f"Stage {i+1}" for i in range(num_stages)]

    # 2. Determine grid size
    num_plots = len(valid_keys)
    cols = 2
    rows = (num_plots + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Handle the case where there is only one plot (axes won't be an array)
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. Plotting Loop
    for i, key in enumerate(valid_keys):
        values = data[key]

        # Plot a line for every detected stage
        for stage_idx in range(num_stages):
            label = opt_labels[stage_idx] if stage_idx < len(opt_labels) else f"S{stage_idx}"
            axes[i].plot(values[stage_idx, :], label=label)

        # Formatting
        axes[i].set_title(key.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Value')
        axes[i].legend(fontsize='small', loc='best')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('comprehensive_trajectory_analysis.png')
    print(f"Visualization saved with {num_stages} stages across {num_plots} metrics.")

# Run the plotting function
plot_all_metrics(data, keys_to_visualize)
