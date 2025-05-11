import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, models
from PIL import Image
import pyrealsense2 as rs  # (If not using RealSense, you can remove this import)
import time
import glob
import os
import pytorch_lightning as pl
import timm
import matplotlib.pyplot as plt
import serial  # Added for Arduino serial communication

# -------------------------------
# Define device before usage
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------
# Placeholder Definitions
# ---------------------------------------------------------------
class Denorm:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return x

def meanIoU(y_hat, y):
    return 0.0

# ---------------------------------------------------------------
# 1) DEEPLAB SEGMENTATION MODEL (HandSegModel)
# ---------------------------------------------------------------
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

# Load HandSegModel checkpoint
checkpoint_path = r"C:\Users\Karthik Narayanan\Desktop\Dexhand_data\checkpoint\checkpoint.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
seg_model = HandSegModel(in_channels=3)
seg_model.load_state_dict(checkpoint["state_dict"], strict=False)
seg_model.eval()
seg_model.to(device)
print("HandSegModel loaded successfully!")

# ---------------------------------------------------------------
# 2) TRANSFORMS & HELPER FUNCTIONS FOR SEGMENTATION
# ---------------------------------------------------------------
seg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_for_segmentation(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    t3d = seg_transform(pil_img)         # [3,224,224]
    return t3d.unsqueeze(0).to(device)     # [1,3,224,224]

def segment_frame(frame_bgr: np.ndarray):
    try:
        inp = preprocess_for_segmentation(frame_bgr)
        with torch.no_grad():
            out = seg_model(inp)  # shape: [1,2,224,224]
            out_np = torch.sigmoid(out).squeeze(0).cpu().numpy()  # [2,224,224]
        mask_small = (out_np[1] > 0.5).astype(np.uint8)
        h, w = frame_bgr.shape[:2]
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 2] = 255  # Blue channel
        overlayed = np.where(mask[..., None] == 1, 0.5 * frame_bgr + 0.5 * overlay, frame_bgr).astype(np.uint8)
        return overlayed, mask
    except Exception as e:
        print(f"Segmentation Error: {e}")
        return frame_bgr, np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

def hsv_filter(frame_bgr: np.ndarray):
    hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    hand_mask = cv2.bitwise_or(mask1, mask2)
    hand_extracted = cv2.bitwise_and(frame_bgr, frame_bgr, mask=hand_mask)
    return hand_extracted, hand_mask

DATA_IMAGE_SIZE = (224, 224)
final_transform = transforms.Compose([
    transforms.Resize(DATA_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def preprocess_for_dataset(hand_bgr: np.ndarray):
    rgb_img = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return final_transform(pil_img)

# ---------------------------------------------------------------
# 3) CLASSIFIER MODEL DEFINITION & LOADING (Joint Angle Prediction)
# ---------------------------------------------------------------
NUM_JOINTS = 15
NUM_BINS = 10

class MobileNetV2HandClassifier(nn.Module):
    def __init__(self, num_joints=NUM_JOINTS, num_bins=NUM_BINS):
        super().__init__()
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.model = timm.create_model(
            "mobilenetv2_100", pretrained=False, in_chans=9, num_classes=num_joints * num_bins
        )
    def forward(self, x):
        out = self.model(x)  # shape: (batch, num_joints*num_bins)
        out = out.view(-1, self.num_joints, self.num_bins)  # reshape to (batch, 15, 10)
        return out

classifier_model = MobileNetV2HandClassifier().to(device)
classifier_model.load_state_dict(torch.load("mobilenetv2_hand_pose_classification.pth", map_location=device))
classifier_model.eval()
print("MobileNetV2HandClassifier loaded successfully!")

# ---------------------------------------------------------------
# 4) FRONT + WEBCAM SETUP (Using Front Webcam for Front View)
# ---------------------------------------------------------------
FRONT_CAM_ID = 1  # Front webcam replacing RealSense
RIGHT_CAM_ID = 3
LEFT_CAM_ID  = 2

front_cam = cv2.VideoCapture(FRONT_CAM_ID)
right_cam = cv2.VideoCapture(RIGHT_CAM_ID)
left_cam  = cv2.VideoCapture(LEFT_CAM_ID)

if not front_cam.isOpened():
    print("Error: Could not open Front Camera.")
if not right_cam.isOpened():
    print("Error: Could not open Right Camera.")
if not left_cam.isOpened():
    print("Error: Could not open Left Camera.")

print("Cameras ready. Press 'q' to quit.")

# ---------------------------------------------------------------
# Arduino Serial Communication Setup
# ---------------------------------------------------------------
try:
    ser = serial.Serial('COM6', 115200, timeout=5)  # Update COM port as needed
    print("Arduino serial connection established!")
except Exception as e:
    print(f"Error: Could not open Arduino serial connection: {e}")
    ser = None

# ---------------------------------------------------------------
# 5) OPTIONAL: SETUP MATPLOTLIB FOR LIVE BAR CHART DISPLAY
# ---------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

last_seg_time = time.time()
seg_interval = 4.0
last_formatted_serial = None  # Stores the most recent formatted serial message

while True:
    # Acquire frames from the three cameras
    ret_front, front_bgr = front_cam.read()
    ret_right, right_bgr = right_cam.read()
    ret_left, left_bgr = left_cam.read()

    if not ret_front or not ret_right or not ret_left:
        print("Error: Could not read from one or more cameras.")
        continue

    now = time.time()
    # Every seg_interval seconds, update the segmentation, classification, and serial message
    if (now - last_seg_time) >= seg_interval:
        last_seg_time = now

        # -------------------------------
        # SEGMENT each frame using the segmentation model
        # -------------------------------
        seg_front, _ = segment_frame(front_bgr)
        seg_right, _ = segment_frame(right_bgr)
        seg_left,  _ = segment_frame(left_bgr)

        # Apply HSV filter to each segmented image
        hsv_front, _ = hsv_filter(seg_front)
        hsv_right, _ = hsv_filter(seg_right)
        hsv_left,  _ = hsv_filter(seg_left)

        # Replicate dataset transform for classifier input:
        front_tensor = preprocess_for_dataset(hsv_front)
        right_tensor = preprocess_for_dataset(hsv_right)
        left_tensor  = preprocess_for_dataset(hsv_left)

        # Combine into a 9-channel tensor [1,9,224,224]
        combined_9ch = torch.cat([front_tensor, right_tensor, left_tensor], dim=0).unsqueeze(0).to(device)

        # -------------------------------
        # CLASSIFICATION: Predict Joint Angles
        # -------------------------------
        with torch.no_grad():
            logits = classifier_model(combined_9ch)  # shape: [1,15,10]
            pred_classes = torch.argmax(logits, dim=2).cpu().numpy()[0]
        pred_angles = 90 + pred_classes * 10
        print("Predicted Joint Angles:", pred_angles)

        # ------------------------------------------------
        # Process predicted angles into serial format
        # ------------------------------------------------
        # Convert predicted angles using the formula (angle - 90) * 2.
        # For example, 90 -> 0, 100 -> 20, 110 -> 40, etc.
        converted_angles = ((pred_angles - 90) * 2).astype(int)
        serial_angles = []

        # Process the four fingers (Index, Middle, Ring, Pinky)
        # Predicted order: indices 3,4,5 (Index), 6,7,8 (Middle), 9,10,11 (Ring), 12,13,14 (Pinky)
        for i in range(4):
            start_idx = 3 + i * 3
            mcp = converted_angles[start_idx]       # MCP angle for the finger
            pip = converted_angles[start_idx + 1]     # PIP angle for the finger
            # Serial format for each finger: [MCP, MCP, PIP]
            serial_angles.extend([mcp, mcp, pip])

        # Process the thumb (predicted indices 0,1,2)
        thumb_angles = converted_angles[0:3].tolist()
        # Move the thumb values to the end
        serial_angles.extend(thumb_angles)

        # Append the fixed value 120 as the last entry
        serial_angles.append(120)

        print("Serial Format before index-specific formatting:", serial_angles)

        # ------------------------------------------------
        # Further format the serial angles based on index-specific rules:
        # For index 0, keep as is
        # For index 1, 180 - angle
        # For index 2, 180 - angle
        # For index 3, keep as is
        # For index 4, 180 - angle
        # For index 5, 180 - angle
        # For index 6, 180 - angle
        # For index 7, keep as is
        # For index 8, keep as is
        # For index 9, 180 - angle
        # For index 10, keep as is
        # For index 11, keep as is
        # For index 12, 180 - angle
        # For index 13, keep as is
        # For index 14, 180 - angle
        # For index 15, keep as is (fixed value 120)
        # ------------------------------------------------
        indices_to_flip = {1, 2, 4, 5, 6, 9, 12, 14}
        formatted_serial = []
        for i, angle in enumerate(serial_angles):
            if i in indices_to_flip:
                formatted_serial.append(180 - angle)
            else:
                formatted_serial.append(angle)

        print("Final Serial Format:", formatted_serial)

        # Store the new formatted serial message for continuous sending
        last_formatted_serial = formatted_serial

        # -------------------------------
        # Update live bar chart with predictions
        # -------------------------------
        ax.cla()
        ax.bar(np.arange(NUM_JOINTS) - 0.2, pred_angles, width=0.4,
               label="Predicted Angles (°)", color='blue', alpha=0.9, edgecolor='black')
        ax.bar(np.arange(NUM_JOINTS) + 0.2, pred_classes * 10 + 90, width=0.4,
               label="Predicted Bins", color='red', alpha=0.9, edgecolor='black')
        ax.set_xticks(np.arange(NUM_JOINTS))
        ax.set_xticklabels([f"Joint {i}" for i in range(NUM_JOINTS)], fontsize=10, rotation=0)
        ax.set_ylabel("Angle (°)", fontsize=12, fontweight='bold')
        ax.set_ylim(80, 190)
        ax.set_title("Predicted Joint Angles from Hand Pose", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Overlay predicted angles on the segmented front image
        overlay_text = ", ".join([f"J{i}:{angle}" for i, angle in enumerate(pred_angles)])
        disp_front = seg_front.copy()
        cv2.putText(disp_front, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Use segmented images for right and left views
        disp_right = seg_right
        disp_left  = seg_left

    else:
        # Not time to re-segment/classify; show raw images
        disp_front = front_bgr
        disp_right = right_bgr
        disp_left  = left_bgr

    # -------------------------------
    # Continuously send the last formatted serial message to Arduino
    # -------------------------------
    if last_formatted_serial is not None:
        serial_data = "[" + ", ".join(str(angle) for angle in last_formatted_serial) + "]"
        print("Serial message to Arduino:", serial_data)  # Debug print
        if ser is not None and ser.is_open:
            ser.write(serial_data.encode())
            ser.flush()  # Ensure the message is sent out

    # -------------------------------
    # Display the images
    # -------------------------------
    top_row = np.hstack([disp_front, disp_right, disp_left])
    cv2.imshow("3-Cams: Seg+HSV & Joint Angle Prediction", top_row)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Small delay to control the continuous sending rate
    time.sleep(0.1)


# ---------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------
front_cam.release()
right_cam.release()
left_cam.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
print("Exited.")
