import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory data
data = np.load('example_datasets/processed/arctic/leap_hand/right/notebook-26-66-s07-u02/0/trajectory_dexmachina.npz')

# Labels for the 4 optimization stages based on your timesteps
opt_labels = ["10/40", "20/40", "30/40", "40/40"]

# --- EASY CONFIGURATION ---
# Simply add or remove key names from this list to update the plot
keys_to_visualize = [
    'rew_mean',           # Total reward
    'obj_dist_rew_mean',  # Progress toward object
    'obj_arti_rew_mean',  # Success in articulation
    'contact_rew_mean',   # Physical contact quality
    'obj_pos_dist_mean',  # Distance error (lower is usually better)
    'obj_quat_dist_mean', # Rotation error
    'imi_rew_mean',
    'improvement',       # Learning progress
    'fingertip_dist_right_mean',  # Distance of right hand fingertips to object
]

def plot_all_metrics(data, keys, labels):
    # Determine grid size (2 columns)
    num_keys = len(keys)
    cols = 2
    rows = (num_keys + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        if key not in data:
            print(f"Key '{key}' not found in the file. Skipping...")
            continue

        # Get the (4, 32) array for the current metric
        values = data[key]
        print(f"Plotting '{key}' with shape {values.shape}")
        # Plot a line for each of the 4 optimization stages
        for stage_idx in range(values.shape[0]):
            axes[i].plot(values[stage_idx, :], label=f'Opt {labels[stage_idx]}')

        # Clean up titles and labels
        axes[i].set_title(key.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Value')
        axes[i].legend(fontsize='small', loc='best')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Remove any extra empty subplots if the number of keys is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('comprehensive_trajectory_analysis.png')
    print("Visualization saved as 'comprehensive_trajectory_analysis.png'")

# Run the plotting function
plot_all_metrics(data, keys_to_visualize, opt_labels)
