import numpy as np
from enum import Enum
import torch, pygame, os
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_colors = [
    [0.1, .7, 0.2],  # 0 land
    [.65, .27, .07],  # 1 obstable
    [0.3, 0.2, .8],  # 2 player
    [.9, .87, .02],  # 4 reward
    [.8, 0.2, 0.3],  # 3 npc
]
color_maps = [tuple(np.round(np.array(c) * 255).astype(int)) for c in _colors]

class OBJECTS(Enum):
    land = 0
    obstable = 1
    player = 2
    reward = 3
    npc = 4

def plot_progress(title, steps, update_indices):
    steps = np.array(steps)
    x = list(range(len(steps)))

    # Plot the data
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(x, steps, label='steps', color='green', linestyle='-')
    plt.scatter(update_indices, steps[update_indices], s=30, color='red', label='target update', zorder=5)
    plt.title(title)

    # Add labels, title, and legend
    plt.xlabel('Episodes')
    plt.legend()

    # Save the plot as an image
    img_name = f"plots/{len(os.listdir('plots'))}.png"
    plt.savefig(img_name, dpi=300)  # Save with high resolution
    plt.close()
    print('Plot saved to', img_name)




# def plot_progress_with_map(img_name, title, x_label, y_label, color, steps, epsilons, to_scatter, map_screenshot):
# def plot_progress_with_map(img_name, title, x_label, y_label, to_plot, to_scatter, map_screenshot):
#     # Determine aspect ratio of the map screenshot
#     screenshot_aspect = map_screenshot.shape[1] / map_screenshot.shape[0]

#     # Create a figure with two subplots, adjusting width for the screenshot's aspect ratio
#     _, axs = plt.subplots(
#         1, 2,
#         figsize=(8 + 8 * screenshot_aspect, 6),  # Reduced extra width
#         gridspec_kw={'width_ratios': [screenshot_aspect, 1]}  # Adjust subplot widths
#     )

#     # Plot the map (screenshot)
#     axs[0].imshow(map_screenshot)
#     axs[0].axis('off')  # Remove axis for the map
#     axs[0].set_title("Map", fontsize=14)

#     # Plot the progress
#     for p in to_plot:
#         x, y, c, l = p
#         axs[1].plot(x, y, label=l, color=c, linestyle='-')

#     for u_i in to_scatter:
#         x, y, s, c, l = u_i
#         axs[1].scatter(x, y, s=s, color=c, label=l)
#     axs[1].set_title(title, fontsize=14)
#     axs[1].set_xlabel(x_label, fontsize=12)
#     axs[1].set_ylabel(y_label, fontsize=12)
#     axs[1].legend(fontsize=10)

#     # Adjust layout to minimize whitespace
#     plt.tight_layout()

#     # Save the combined plot as an image
#     os.makedirs("plots", exist_ok=True)
    
#     plt.savefig(img_name, dpi=300)
#     plt.close()
#     print('Combined plot saved to', img_name)


def plot_progress_with_map(img_name, x_label_1, y_label_1, x_label_2, y_label_2, 
                           to_plot_1, to_scatter_1, 
                           to_plot_2, to_scatter_2, 
                           map_screenshot):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os
    import numpy as np

    # Determine aspect ratio of the map screenshot
    screenshot_aspect = map_screenshot.shape[1] / map_screenshot.shape[0]

    # Create a figure with gridspec layout
    fig = plt.figure(figsize=(12 + 6 * screenshot_aspect, 8))  # Adjust overall figure size
    gs = GridSpec(2, 2, width_ratios=[screenshot_aspect, 1])  # 2 rows, 2 columns layout

    # Add the map subplot (A)
    ax_map = fig.add_subplot(gs[:, 0])  # Spans both rows in the first column
    ax_map.imshow(map_screenshot)
    ax_map.axis('off')  # Remove axis for the map
    ax_map.set_title("Map", fontsize=14)

    # Add the first graph subplot (C)
    ax_plot1 = fig.add_subplot(gs[0, 1])  # Top row, second column
    for p in to_plot_1:
        x, y, c, l = p
        ax_plot1.plot(x, y, label=l, color=c, linestyle='-')
    for u_i in to_scatter_1:
        x, y, s, c, l = u_i
        ax_plot1.scatter(x, y, s=s, color=c, label=l)
    ax_plot1.set_xlabel(x_label_1, fontsize=12)
    ax_plot1.set_ylabel(y_label_1, fontsize=12)
    ax_plot1.legend(fontsize=10)

    # Add the second graph subplot (D)
    ax_plot2 = fig.add_subplot(gs[1, 1])  # Bottom row, second column
    for p in to_plot_2:
        x, y, c, l = p
        ax_plot2.plot(x, y, label=l, color=c, linestyle='-')
    for u_i in to_scatter_2:
        x, y, s, c, l = u_i
        ax_plot2.scatter(x, y, s=s, color=c, label=l)
    ax_plot2.set_xlabel(x_label_2, fontsize=12)
    ax_plot2.set_ylabel(y_label_2, fontsize=12)
    ax_plot2.legend(fontsize=10)

    # Adjust layout to minimize whitespace
    plt.tight_layout()

    # Save the combined plot as an image
    os.makedirs("plots", exist_ok=True)
    plt.savefig(img_name, dpi=300)
    plt.close()
    print('Combined plot saved to', img_name)

def normalized_image(img):

  mn = img.min()
  img = img - mn
  img = img / img.max()
  img = img * 255
  img = np.array(img).astype(int)

  return img

def create_video(frames, output_path, fps):
    _, height, width, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Ensure frame is in uint8 format and write it
        frame = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

