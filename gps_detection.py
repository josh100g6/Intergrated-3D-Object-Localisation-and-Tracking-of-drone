"""
This program performs object detection for people and then determines their GPS coordinates.
Now includes automated depth estimation using MiDaS.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import projection_utils as pu
import pymap3d as pm
from scipy.spatial.transform import Rotation as R
from transformers import pipeline
import torch
from PIL import Image

# Initialize MiDaS depth estimation model
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

def drawBoundingBox(image, detection):
    x1, y1, x2, y2 = detection.bbox.to_xyxy()
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    cv2.putText(image, detection.category.name, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)
    return image

def estimate_depth(image_path):
    # Load image and convert to PIL format
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)  # Convert to PIL image
    depth_output = depth_estimator(pil_image)  # Pass PIL image to depth estimator
    depth_map = depth_output["depth"]  # Depth map as a numpy array (relative depth)

    # Normalize depth map to a reasonable range (e.g., 0 to 50 meters)
    depth_map = np.array(depth_map)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize to 0-1
    depth_map = depth_map * 50  # Scale to a realistic range (e.g., 0-50 meters)
    
    return depth_map

def get_depth_at_pixel(depth_map, x, y):
    # Ensure coordinates are within bounds
    h, w = depth_map.shape
    x = min(max(int(x), 0), w-1)
    y = min(max(int(y), 0), h-1)
    return depth_map[y, x]

def pixelToGPS(x, y, depth_map):
    xy = np.array([[x], [y]], dtype='float64')
    
    xy_undistorted = cv2.undistortPointsIter(xy, intr_mat, dist_coeffs, None, intr_mat, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 0))
    
    # Get depth (Zc) for the specific pixel
    Zc = get_depth_at_pixel(depth_map, x, y)
    if Zc <= 0:  # Ensure depth is positive
        Zc = 1.0  # Default to a small positive value to avoid division issues
    
    p = Zc * np.array([[xy_undistorted[0][0][0]], [xy_undistorted[0][0][1]], [1]], dtype='float64')

    Pc = np.linalg.inv(intr_mat) @ p
    
    R_C_EDN = R.from_euler('zxy', [drone_angles[2], drone_angles[1], drone_angles[0]], degrees=True).as_matrix()
    e, d, n = R_C_EDN @ Pc
    
    pixel_gps = pm.ned2geodetic(n, e, d, *drone_gps)
    pixel_gps = [pixel_gps[0][0], pixel_gps[1][0], pixel_gps[2][0]]
    
    return pixel_gps, Zc

# Load intrinsic parameters and drone metadata
intr_xml_path = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\Code\Joshua_Honour_Projects\camera Calibration\Drone_4\video\intrinsic\intr_4_video_zoom.xml"
image_path = r"camera Calibration/Drone_4/video/extrinsic/first_frame.jpg"
srt_file = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\DJI-Mavic3E-Drone4-20250509T044838Z-001\DJI-Mavic3E-Drone4\DJI_202412181012_007\DJI_20241218103030_0001_V.SRT"

drone_gps, drone_angles = pu.getVideoFrameMetadata(srt_file, 1)
intr_mat, dist_coeffs = pu.getIntrinsicData(intr_xml_path)

print(f'Drone GPS: {drone_gps}')
print(f'Drone Angles: {drone_angles}')

# Load image and compute depth map
image = cv2.imread(image_path)
depth_map = estimate_depth(image_path)

# Perform object detection
model = YOLO("yolo11n.pt")
model.fuse()

# Lower the confidence threshold to increase detection sensitivity
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics", 
    model=model, 
    confidence_threshold=0.1,  # Reduced from 0.3 to 0.1
    device="cuda:0"
)

# Adjust slice dimensions to match image size (optional: can be tuned based on image resolution)
detections = get_sliced_prediction(
    image_path, 
    detection_model, 
    slice_height=960, 
    slice_width=960
)

# Debug: Print the number of detections
print(f"Total detections: {len(detections.object_prediction_list)}")
print(f"Person detections: {sum(1 for detection in detections.object_prediction_list if detection.category.name == 'person')}")

# Process detections
person_detected = False
for detection in detections.object_prediction_list:
    if detection.category.name == 'person':
        person_detected = True
        image = drawBoundingBox(image, detection)

        x, y, w, h = detection.bbox.to_xywh()
        object_gps, Zc = pixelToGPS(x + w/2, y + h, depth_map)
        
        object_gps_str = f'GPS: {object_gps[0]:.6f}, {object_gps[1]:.6f}, Depth: {Zc:.2f}m'
        cv2.putText(
            image, 
            object_gps_str, 
            (int(x), int(y + h + 20)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, 
            color=(255, 0, 0), 
            thickness=1
        )

# If no people are detected, add a message to the image
if not person_detected:
    cv2.putText(
        image, 
        "No people detected in this frame", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.0, 
        color=(0, 0, 255), 
        thickness=2
    )

# Resize the image for display (optional: adjust scale factor as needed)
scale_factor = 0.5  # Adjust this to fit your screen (e.g., 0.5 reduces size by 50%)
display_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

# Display the result in a resizable window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.imshow('image', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

