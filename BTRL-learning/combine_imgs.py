import natsort
import os
from PIL import Image

unity_screenshot_dir = "/home/finn/repos/MORL-BT/Assets/Screenshots"
runs_dir = "/home/finn/repos/MORL-BT/BTRL-learning/runs/"
exp_name = "flat-acc-button_fetch_trigger/2024-07-10-15-36-00_view_TrainedAgainModel_widerPlotRange"
q_img_dir = runs_dir + exp_name + "/imgs/Q"
acc_img_dir = runs_dir + exp_name + "/imgs/ACC"

combined_img_dir = runs_dir + exp_name + "/imgs/combined_imgs"
os.makedirs(combined_img_dir, exist_ok=True)
unity_img_dir = runs_dir + exp_name + "/imgs/unity"
os.makedirs(unity_img_dir, exist_ok=True)

# get images in both deers and sort
unity_files = natsort.natsorted([f for f in os.listdir(unity_screenshot_dir) if f.endswith(".png")])
q_fun_files = natsort.natsorted([f for f in os.listdir(q_img_dir) if f.endswith(".png")])
acc_fun_files = natsort.natsorted([f for f in os.listdir(acc_img_dir) if f.endswith(".png")])

print(f"Found {len(unity_files)} unity images and {len(q_fun_files)} q_fun images")

# combine images
print("Combining images")
for i, (unity_file, q_fun_file, acc_fun_file) in enumerate(zip(unity_files, q_fun_files, acc_fun_files)):
    unity_img = Image.open(os.path.join(unity_screenshot_dir, unity_file))
    unity_img.save(f"{unity_img_dir}/{unity_file}")
    os.remove(os.path.join(unity_screenshot_dir, unity_file))

    q_fun_img = Image.open(os.path.join(q_img_dir, q_fun_file))
    acc_fun_img = Image.open(os.path.join(acc_img_dir, acc_fun_file))

    # combine images as follows: on the left side, the unity image, on the right side at the top the Q-fucntion, on the right side at the bottom the ACC function
    combined_img = Image.new("RGB", (unity_img.width + q_fun_img.width, unity_img.height))
    combined_img.paste(unity_img, (0, 0))
    combined_img.paste(q_fun_img, (unity_img.width, 0))
    combined_img.paste(acc_fun_img, (unity_img.width, q_fun_img.height))

    combined_img.save(f"{combined_img_dir}/{i}.png")
    print(f"Saved combined image {i}.png")
