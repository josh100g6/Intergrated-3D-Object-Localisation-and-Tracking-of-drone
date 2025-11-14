
"""
Use this program for 2D projection testing on videos taken with the drones. There are options to save it as a video or just show playback as the program runs. 
There is also an option to draw ground truths on selected frames and save relevant data to a CSV to determine potential trends causing the offset issues. The code to select ground truth points is the same as that from the mixed_calibration_images.py program. If you don't know how to use that, read the top block comment in that file. The image you will select ground truth points on will have the projected points already on it. Make sure to select ground truth points in the order that correspond to the projected ones. If a frame doesn't have the chessboard in it or it is only partially in it, just skip it, it won't save that frame's data. If you are happy with the data you have obtained but there are still more frames, just force close the program. The obtained data is already saved, but the temp folder will need to be manually deleted if you care about that.
"""
import cv2
import projection_utils as pu
import Manual_Calibration as mci
import shutil
import os
import pandas as pd
import numpy as np
# Set save_vid to True if you want to save the projection video.
save_vid = False
# Set show_playback to True to show the projected points on each frame as the program runs.
show_playback = True
# Set get_gts to True if you want to select the ground truth points.
get_gts = True
# Set save_gt_images to True if you want to save the frames with the ground truths on them.
save_gt_images = True
# Path to the SRT file for the MP4 containing the frame used for the initial extrinsic calibration.
srt_file0 = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\DJI-Mavic3E-Drone4-20250509T044838Z-001\DJI-Mavic3E-Drone4\DJI_202501171525_017\DJI_20250117153008_0002_V.SRT"
# Path to the initial extrinsic XML file.
init_extr_xml_path = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\Code\Joshua_Honour_Projects\camera Calibration\Drone_4\video\extrinsic\extr_0_moving.xml"
# Path to the MP4 and SRT files (excluding the file extension) you want to project onto.
path1 = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\DJI-Mavic3E-Drone4-20250509T044838Z-001\DJI-Mavic3E-Drone4\DJI_202501171525_017\DJI_20250117153008_0002_V"
mp4_file1 = f"{path1}.MP4"
srt_file1 = f"{path1}.SRT"
# Path to the intrinsic XML file for the camera used to take the video.
intr_xml_path = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\Code\Joshua_Honour_Projects\camera Calibration\Drone_4\video\intrinsic\intr_4_video_zoom.xml"
# Grid configuration
dim = (9,6)  # Matches 8x6 squares (9x7 internal corners)
sq_size = 0.5  # Square size in meters, adjusted to 30 cm based on typical drone calibration chessboard
grid_origin = [0.0,0.0, 0.0]  # Adjustable origin [x, y, z] in meters
rotation_angle_deg = 0  # Manual rotation adjustment to fix -90 degree misalignment
# The frame the video starts at if you have modified the MP4. If you haven't modified the MP4 put the starting frame as 1.
starting_frame = 1
# The amount of frames to skip. For projection I use 1 to process and show every frame. For the ground truth selection I chose 30.
frame_increment = 30
# Name of the folder to save ground truth images and the CSV file to.
gt_folder = 'test'
# Display resolution for playback and ground truth images
display_width = 1280
display_height = 720
def resize_frame(frame, width, height):
    """
    Resize frame to specified width and height while preserving aspect ratio.
    Returns resized frame and scale factors (sx, sy).
    """
    orig_height, orig_width = frame.shape[:2]
    aspect_ratio = orig_width / orig_height
    if aspect_ratio > width / height:
        new_width = width
        new_height = int(width / aspect_ratio)
    else:
        new_height = height
        new_width = int(height * aspect_ratio)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    sx = orig_width / new_width
    sy = orig_height / new_height
    return resized, sx, sy
def apply_rotation_to_points(points, angle_deg):
    """
    Rotate 3D points around Z-axis by angle_deg (degrees).
    """
    angle_rad = np.radians(angle_deg)
    Rz = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                   [np.sin(angle_rad), np.cos(angle_rad), 0],
                   [0, 0, 1]])
    rotated_points = points @ Rz.T
    return rotated_points
def to_homogeneous_matrix(mat):
    """
    Convert to 4x4 homogeneous matrix if necessary.
    """
    if mat.shape == (3, 4):
        homog_mat = np.eye(4)
        homog_mat[:3, :] = mat
        return homog_mat
    elif mat.shape == (4, 4):
        return mat
    else:
        raise ValueError(f"Unexpected matrix shape: {mat.shape}. Expected (3,4) or (4,4)")
# Verify file existence
for path in [srt_file0, init_extr_xml_path, mp4_file1, srt_file1, intr_xml_path]:
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit()
try:
    gps0, angles0 = pu.getVideoFrameMetadata(srt_file0, starting_frame)
except (FileNotFoundError, ValueError) as e:
    print(f"Error extracting metadata from SRT file0: {e}")
    exit()
print(f'Drone 0: {gps0}, {angles0}\n')
try:
    intr_mat, dist_coeffs = pu.getIntrinsicData(intr_xml_path)
except ValueError as e:
    print(f"Error loading intrinsic parameters: {e}")
    exit()
# Recalibrate initial extrinsic matrix using the first frame
cap = cv2.VideoCapture(mp4_file1)
if not cap.isOpened():
    print(f"Error: Could not open video file {mp4_file1}")
    exit()
ret, frame = cap.read()
if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    checkerboard_size = (9, 6)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= sq_size
    # Apply rotation first to ensure correct orientation during recalibration
    objp = apply_rotation_to_points(objp, rotation_angle_deg)
    # Then apply origin offset
    objp += grid_origin
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        ret, rvec, tvec = cv2.solvePnP(objp, corners, intr_mat, dist_coeffs)
        if ret:
            rmat, _ = cv2.Rodrigues(rvec)
            init_extr = np.eye(4)
            init_extr[:3, :3] = rmat
            init_extr[:3, 3] = tvec.flatten()
            fs = cv2.FileStorage(init_extr_xml_path, cv2.FILE_STORAGE_WRITE)
            fs.write("extrinsic_matrix", init_extr)
            fs.release()
            print(f"Recalibrated initial extrinsic matrix saved to {init_extr_xml_path}")
        else:
            print("PnP failed. Using existing init_extr.")
    else:
        print("Chessboard not detected in first frame. Using existing init_extr.")
else:
    print("Failed to read first frame. Using existing init_extr.")
try:
    init_extr = pu.getInitialExtrinsicMatrix(init_extr_xml_path)
    init_extr = to_homogeneous_matrix(init_extr)
    print(f"Initial Extrinsic Shape: {init_extr.shape}")
except ValueError as e:
    print(f"Error loading or converting initial extrinsic matrix: {e}")
    exit()
cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame - 1)
if save_vid:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
else:
    frame_width = display_width
    frame_height = display_height
if get_gts:
    if not os.path.exists(f'{gt_folder}/'):
        os.mkdir(f'{gt_folder}/')
        os.mkdir(f'{gt_folder}/temp/')
    elif not os.path.exists(f'{gt_folder}/temp/'):
        os.mkdir(f'{gt_folder}/temp/')
frame_count = 0
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        try:
            gps1, angles1 = pu.getVideoFrameMetadata(srt_file1, frame_count + starting_frame)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error extracting metadata from SRT file1 for frame {frame_count+starting_frame}: {e}")
            frame_count += frame_increment
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            continue
        t_edn = pu.getEDNTranslation(gps0, gps1)
        rel_extr = pu.getRelativeExtrinsicMatrix(angles0, angles1, t_edn)
        rel_extr = to_homogeneous_matrix(rel_extr)
        print(f"Relative Extrinsic Shape: {rel_extr.shape}")
        print(f"Frame {frame_count + starting_frame}:")
        print(f"Angles0: {angles0}, Angles1: {angles1}")
        print(f"Relative Extrinsic Rotation Matrix:\n{rel_extr[:3, :3]}\n")
        # Define and adjust 3D points
        grid_w, grid_h = dim
        obj_points = np.zeros((grid_w * grid_h, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:grid_w, 0:grid_h].T.reshape(-1, 2)
        obj_points *= sq_size
        # Apply rotation first
        obj_points = apply_rotation_to_points(obj_points, rotation_angle_deg)
        # Then apply origin offset
        obj_points += grid_origin
        extr = init_extr @ rel_extr
        print(f"Combined Extrinsic Shape: {extr.shape}")
        print(f"3D Points (first 5 after rotation and offset):\n{obj_points[:5]}...")
        img_points, _ = cv2.projectPoints(obj_points, extr[:3, :3], extr[:3, 3], intr_mat, dist_coeffs)
        img_points = img_points.squeeze().astype(int)
        for (x, y) in img_points:
            cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        points = img_points.tolist()
        display_frame, sx, sy = resize_frame(frame, display_width, display_height)
        if save_vid:
            out.write(frame)
        elif show_playback:
            cv2.imshow("Playback", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if get_gts:
            cv2.imwrite(f"{gt_folder}/temp/frame.jpg", display_frame)
            point_selector = mci.PointSelector(f"{gt_folder}/temp/", 1)
            corners = [None]
            corners, _ = point_selector.run(0, 'intrinsic', dim)
            if corners is not None:
                corners = corners.squeeze().astype(float)
                corners[:, 0] *= sx
                corners[:, 1] *= sy
                corners = corners.astype(int).tolist()
                if save_gt_images:
                    for x, y in corners:
                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                    save_frame, _, _ = resize_frame(frame, display_width, display_height)
                    cv2.imwrite(f"{gt_folder}/frame{frame_count+starting_frame}.jpg", save_frame)
                # Auto-compute offset for the first frame
                if frame_count == 0 and len(corners) == len(points):
                    proj_array = np.array(points)
                    gt_array = np.array(corners)
                    offset_x = np.mean(gt_array[:, 0] - proj_array[:, 0])
                    offset_y = np.mean(gt_array[:, 1] - proj_array[:, 1])
                    print(f"Auto-computed offset (pixels): x = {offset_x:.2f}, y = {offset_y:.2f}")
                    # Convert to meters (adjust scale_factor based on image resolution)
                    scale_factor = 0.001  # Example, adjust based on your image resolution
                    grid_origin[0] += offset_x * scale_factor
                    grid_origin[1] += offset_y * scale_factor
                    print(f"Updated grid origin: {grid_origin}")
                    # Recompute points with updated origin
                    obj_points = np.zeros((grid_w * grid_h, 3), np.float32)
                    obj_points[:, :2] = np.mgrid[0:grid_w, 0:grid_h].T.reshape(-1, 2)
                    obj_points *= sq_size
                    obj_points = apply_rotation_to_points(obj_points, rotation_angle_deg)
                    obj_points += grid_origin
                    img_points, _ = cv2.projectPoints(obj_points, extr[:3, :3], extr[:3, 3], intr_mat, dist_coeffs)
                    img_points = img_points.squeeze().astype(int)
                    points = img_points.tolist()
                    # Redraw points with updated positions
                    frame_copy = frame.copy()
                    for (x, y) in points:
                        cv2.circle(frame_copy, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                    display_frame, sx, sy = resize_frame(frame_copy, display_width, display_height)
                    cv2.imshow("Playback", display_frame)
                    cv2.waitKey(1)
                temp = gps0 + gps1 + angles0 + angles1 + [dim[0]*dim[1]] + corners + points
                df = pd.DataFrame([temp])
                df.to_csv(f"{gt_folder}/data.csv", index=False, header=False, mode='a')
        frame_count += frame_increment
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
cap.release()
if save_vid:
    out.release()
elif get_gts:
    shutil.rmtree(f'{gt_folder}/temp/')
cv2.destroyAllWindows()
