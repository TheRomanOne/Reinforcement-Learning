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



def capture_screenshot_as_array(surface):
    """Capture the pygame screen as a NumPy array."""
    screenshot = pygame.surfarray.array3d(surface)  # Shape: (width, height, 3)
    screenshot = np.transpose(screenshot, (1, 0, 2))  # Transpose to (height, width, 3)
    return screenshot

def plot_progress_with_map(img_name, title, steps, epsilons, update_indices, map_screenshot):
    steps = np.array(steps)
    epsilons = np.array(epsilons)

    # Determine aspect ratio of the map screenshot
    screenshot_aspect = map_screenshot.shape[1] / map_screenshot.shape[0]

    # Create a figure with two subplots, adjusting width for the screenshot's aspect ratio
    _, axs = plt.subplots(
        1, 2,
        figsize=(8 + 8 * screenshot_aspect, 6),  # Reduced extra width
        gridspec_kw={'width_ratios': [screenshot_aspect, 1]}  # Adjust subplot widths
    )

    # Plot the map (screenshot)
    axs[0].imshow(map_screenshot)
    axs[0].axis('off')  # Remove axis for the map
    axs[0].set_title("Map", fontsize=14)

    # Plot the progress
    axs[1].plot(epsilons, steps, label='steps', color='green', linestyle='-')
    axs[1].scatter(epsilons[update_indices], steps[update_indices], s=10, color='red', label='target update')
    axs[1].invert_xaxis()
    axs[1].set_title(title, fontsize=14)
    axs[1].set_xlabel('Epsilon', fontsize=12)
    axs[1].set_ylabel('Steps', fontsize=12)
    axs[1].legend(fontsize=10)

    # Adjust layout to minimize whitespace
    plt.tight_layout()

    # Save the combined plot as an image
    os.makedirs("plots", exist_ok=True)
    
    plt.savefig(img_name, dpi=300)
    plt.close()
    print('Combined plot saved to', img_name)

# def convert_webm_to_mp4(input_path, output_path):
#     # Open the .webm file
    
#     video = cv2.VideoCapture(input_path)
    
#     # Get properties of the input video
#     fps = int(video.get(cv2.CAP_PROP_FPS))
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    
#     # Create a VideoWriter object for the .mp4 output
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         # Write each frame to the .mp4 file
#         out.write(frame)
    
#     # Release resources
#     video.release()
#     out.release()
#     print(f"Converted {input_path} to {output_path}")

# if __name__ == '__main__':
#     # Example usage
#     input_file = "/home/roman/Videos/Screencasts/Screencast from 2024-11-27 19-36-57.webm"
#     output_file = "output.mp4"
#     convert_webm_to_mp4(input_file, output_file)

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

