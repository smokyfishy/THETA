import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import mediapipe as mp
import os
import time
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.utils import make_grid
from torch.optim import Adam
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


NUM_IMAGES = 400  # Number of images per camera
RIGHT_CAM_ID = 2   # Right Webcam index
LEFT_CAM_ID = 3   # Left Webcam index
DATASET_PATH = "gesture_dataset" # Root dataset directory
CSV_PATH = "Gesture_Angles.csv" # CSV file containing angles

# === Load Gesture Angles CSV ===
angles_df = pd.read_csv(CSV_PATH)

# === Get Gesture Number Input ===
gesture_number = input("Enter Gesture Number: ")

# Find the row in CSV with the selected gesture number
gesture_row = angles_df[angles_df["Number"] == int(gesture_number)]

if gesture_row.empty:
    print(f"Gesture {gesture_number} not found in CSV.")
    exit()

# Extract only the joint angle columns
ANGLE_COLUMNS = [
    "TH_CMC", "TH_MCP", "TH_IP",
    "IN_MCP", "IN_PIP", "IN_DIP",
    "MI_MCP", "MI_PIP", "MI_DIP",
    "RI_MCP", "RI_PIP", "RI_DIP",
    "PI_MCP", "PI_PIP", "PI_DIP"
]
base_angles = gesture_row[ANGLE_COLUMNS].values.flatten()  # Ensure it's a flat array

# === Setup Dataset Folder ===
gesture_folder = os.path.join(DATASET_PATH, f"gesture_{gesture_number}")
os.makedirs(gesture_folder, exist_ok=True)

# === Create CSV Files for Storing Data ===
angles_csv_path = os.path.join(gesture_folder, "angles.csv")
landmarks_csv_path = os.path.join(gesture_folder, "landmarks.csv")

# Create empty DataFrames for CSV files
angles_df = pd.DataFrame(columns=["Frame"] + ANGLE_COLUMNS)
landmarks_df = pd.DataFrame(columns=["Frame"] + ANGLE_COLUMNS)  # Using angle names for landmarks

# === Initialize MediaPipe Hands for 2D Joint Detection ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# === Initialize RealSense Depth Camera ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable RGB stream only
pipeline.start(config)



right_cam = cv2.VideoCapture(RIGHT_CAM_ID)
left_cam = cv2.VideoCapture(LEFT_CAM_ID)

if not right_cam.isOpened():
    print("Error: Could not open Right Camera.")
if not left_cam.isOpened():
    print("Error: Could not open Left Camera.")


for i in range(5, 0, -1):
    print(f"Starting in {i} seconds...")
    time.sleep(1)
print("-Collecting Data (Move Hand Only Once)...")


for i in range(NUM_IMAGES):
    # Get RealSense RGB frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Error: Could not get RealSense RGB frame.")
        continue
    depth_img = np.asanyarray(color_frame.get_data())  # Convert to NumPy array

    # Read from right and left webcams
    ret_right, right_img = right_cam.read()
    ret_left, left_img = left_cam.read()

    if not ret_right or not ret_left:
        print("Error: Could not get frames from webcams.")
        continue

    for cam, img, suffix in [("front", depth_img, "front"), ("right", right_img, "right"), ("left", left_img, "left")]:
        raw_image_path = os.path.join(gesture_folder, f"frame_{i:03d}_{suffix}.jpg")
        cv2.imwrite(raw_image_path, img)

    
    frame_angles = [angle + random.uniform(-5, 5) for angle in base_angles]

    
    new_angle_row = pd.DataFrame([[i] + frame_angles], columns=["Frame"] + ANGLE_COLUMNS)
    angles_df = pd.concat([angles_df, new_angle_row], ignore_index=True)

    print(f"Collected Frame {i+1}/{NUM_IMAGES}")


angles_df.to_csv(angles_csv_path, index=False)
print(f"Angles data saved to {angles_csv_path}")


pipeline.stop()
right_cam.release()
left_cam.release()
cv2.destroyAllWindows()
print("Data Collection Complete! Moving to Segmentation...")

class HandSegModel(pl.LightningModule):
    """
    This model is based on the PyTorch DeepLab model for semantic segmentation.
    """
    def __init__(self, pretrained=False, lr=1e-4, in_channels=3):
        super().__init__()
        assert in_channels in [1, 3, 4]
        self.deeplab = self._get_deeplab(pretrained=pretrained, num_classes=2, in_channels=in_channels)
        self.denorm_image_for_tb_log = None # For tensorboard logging
        self.lr = lr
        if pretrained:
            if in_channels == 1:
                mean, std = np.array([0.5]), np.array([0.5]) 
            elif in_channels == 3:
                mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]) 
            elif in_channels == 4:
                mean, std = np.array([0.485, 0.456, 0.406, 0.5]), np.array([0.229, 0.224, 0.225, 0.5]) 
            self.denorm_image_for_tb_log = Denorm(mean, std)

    def _get_deeplab(self, pretrained=False, num_classes=2, in_channels=3):
        """
        Get the PyTorch DeepLab model architecture.
        """
        deeplab = models.segmentation.deeplabv3_resnet50(
            pretrained=False,
            num_classes=num_classes
        )
        if pretrained:
            deeplab_21 = models.segmentation.deeplabv3_resnet50(
                pretrained=True,
                progress=True,
                num_classes=21
            )
            for c1, c2 in zip(deeplab.children(), deeplab_21.children()):
                for p1, p2 in zip(c1.parameters(), c2.parameters()):
                    if p1.shape == p2.shape:
                        p1.data = p2.data
        if in_channels == 1:
            weight = deeplab.backbone.conv1.weight
            deeplab.backbone.conv1.weight = nn.Parameter(weight.data[:, 0:1])
        elif in_channels == 4:
            weight = deeplab.backbone.conv1.weight
            C, _, H, W = weight.shape
            deeplab.backbone.conv1.weight = nn.Parameter(torch.cat([
                weight.data,
                torch.randn(C, 1, H, W, device=weight.device) * 0.1,
            ], 1))
        return deeplab

    def forward(self, x):
        return self.deeplab(x)['out']

    def training_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        y_hat = F.softmax(logits, 1).detach()
        miou = meanIoU(y_hat, y.argmax(1))

        # Cache
        self.log('train_bce', loss, prog_bar=True)
        self.log('train_mIoU', miou, prog_bar=True)
        return loss

    def validation_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        y_hat = F.softmax(logits, 1).detach()
        miou = meanIoU(y_hat, y.argmax(1))

        # Cache
        self.log('validation_bce', loss, prog_bar=True)
        self.log('validation_mIoU', miou, prog_bar=True)
        if idx_batch == 0:
            tb_log = self.trainer.logger.experiment
            if self.denorm_image_for_tb_log:
                x = self.denorm_image_for_tb_log(x)
            x_grid = make_grid(x[:16], nrow=4)
            y_hat_grid = make_grid(y_hat[:16].argmax(1).unsqueeze(1), nrow=4)[0:1]
            tb_log.add_image('validation_images', x_grid.cpu().numpy())
            tb_log.add_image('validation_preds', y_hat_grid.cpu().numpy())
        return loss

    def test_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        y_hat = F.softmax(logits, 1).detach()
        miou = meanIoU(y_hat, y.argmax(1))

        # Cache
        self.log('test_bce', loss, prog_bar=True)
        self.log('test_mIoU', miou, prog_bar=True)
        if idx_batch == 0:
            tb_log = self.trainer.logger.experiment
            if self.denorm_image_for_tb_log:
                x = self.denorm_image_for_tb_log(x)
            x_grid = make_grid(x[:16], nrow=4)
            y_hat_grid = make_grid(y_hat[:16].argmax(1).unsqueeze(1), nrow=4)[0:1]
            tb_log.add_image('test_images', x_grid.cpu().numpy())
            tb_log.add_image('test_preds', y_hat_grid.cpu().numpy())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def set_denorm_fn(self, denorm_fn):
        self.denorm_image_for_tb_log = denorm_fn

# Now load the checkpoint
checkpoint = torch.load(r"C:\Users\Karthik Narayanan\Desktop\Dexhand_data\checkpoint\checkpoint.ckpt", map_location="cpu", weights_only=False)

# Extract model state_dict
state_dict = checkpoint["state_dict"]

# Initialize the model
model = HandSegModel(in_channels=3)

# Load weights into model
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Model loaded successfully!")

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
model.to(device)  #  Move model to GPU


#  Image Preprocessing for Model
def preprocess_image(image_path):
    """Load and preprocess image for segmentation model."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device), image

#  Segmentation Function
def segment_and_save(image_path, output_path):
    """Segments hands from an image and saves the result."""
    try:
        input_tensor, original_image = preprocess_image(image_path)

        with torch.no_grad():
            output_dict = model(input_tensor.to(device))

        #  Extract segmentation mask
        output = torch.sigmoid(output_dict).squeeze(0).cpu().numpy()  #  Move result to CPU for saving

        segmented_mask = (output[1] > 0.5).astype(np.uint8)

        #  Resize mask to match original image
        segmented_mask = cv2.resize(segmented_mask, (original_image.width, original_image.height), interpolation=cv2.INTER_NEAREST)

        #  Convert to 3-channel for overlay
        mask_overlay = np.zeros((original_image.height, original_image.width, 3), dtype=np.uint8)
        mask_overlay[:, :, 2] = 255  # Blue overlay

        #  Blend with original image
        original_np = np.array(original_image)
        overlayed_image = np.where(segmented_mask[:, :, None] == 1, 0.5 * original_np + 0.5 * mask_overlay, original_np).astype(np.uint8)

        #  Save segmented image
        cv2.imwrite(output_path, overlayed_image)


        print(f"Processed & Saved: {output_path}")
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# === Run Segmentation on All Collected Images ===
for i in range(NUM_IMAGES):
    for suffix in ["front", "right", "left"]:
        raw_image_path = os.path.join(gesture_folder, f"frame_{i:03d}_{suffix}.jpg")
        segmented_image_path = os.path.join(gesture_folder, f"frame_{i:03d}_{suffix}_segmented.jpg")

        # Run segmentation and save
        segment_and_save(raw_image_path, segmented_image_path)

print("All images segmented and saved")
