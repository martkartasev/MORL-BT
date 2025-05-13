import os
import cv2
from natsort import natsorted
import argparse

def make_video_from_images(folder_path, output_path, duration_per_frame):
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    # Find all image files in folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        print("No image files found in the folder.")
        return

    # Sort images in natural order
    image_files = natsorted(image_files)

    # Read the first image to get dimensions
    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Failed to read the first image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Calculate frames per second from duration per frame
    fps = 1.0 / duration_per_frame

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_file}, skipping.")
            continue

        # Resize image if needed to match first image size
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))

        video.write(img)

    video.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from images in a folder.")
    parser.add_argument("--duration", type=float, default=0.02, help="Duration per frame in seconds (default: 1.0 sec)")
    args = parser.parse_args()

    base_dir = "/home/finn/repos/MORL-BT/BTRL-learning/views/SimpleAccEnv-wide-withConveyer-unshapedSum-v1/2025-05-07-15-29-02_video_RL/bt_rollouts"
    # base_dir = "/home/finn/repos/MORL-BT/BTRL-learning/views/SimpleAccEnv-wide-withConveyer-goal-v1/2025-05-07-14-50-07_video_BTRL/bt_rollouts"
    # base_dir = "/home/finn/repos/MORL-BT/BTRL-learning/views/SimpleAccEnv-wide-withConveyer-goal-v1/2025-05-06-16-26-59_video_CBTRL/bt_rollouts"
    
    rollout_dirs = list(range(10))
    modality_dirs = ["env", "Unshaped RL"]  # RL
    # modality_dirs = ["env", "Goal", "Safety"]  # BTRL
    # modality_dirs = ["env", "feasibility", "Goal", "Safety"]  # CBTRL

    for rollout_dir in rollout_dirs:
        for modality_dir in modality_dirs:
            img_folder = f"{base_dir}/{rollout_dir}/{modality_dir}"
            output_path = f"{img_folder}/{modality_dir}_fast.mp4"

            make_video_from_images(img_folder, output_path, args.duration)