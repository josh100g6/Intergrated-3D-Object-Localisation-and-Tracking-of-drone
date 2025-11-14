"""
I was provided two separate programs for manual and automatic camera calibration. I modified them to create this program that combines both methods into one and to work directly with images.

Usage:
- Step 1: Choose the calibration mode and provide the required paths in the main section of the code. Then run the program.
- Step 2: Select anywhere on the image window to select the frame. If you wish to skip the current image press 's'.
- Step 3: Press 'a' for automatic calibration or 'm' for manual calibration. Default is automatic. You can switch between them at any time, just make sure to clear any selected points using 'r' when you do.
- Step 4:
    - For automatic: Click in the middle of the 4 corner squares. The order matters. Choose a square as top-left, then go top-right, bottom-right, bottom-left. Below is an example diagram.
        1 > 2
            v  
        4 < 3
        If automatic calibration fails, the selected points will disappear and you will need to reselect the window to try again. If you can't get automatic calibration to work just switch to manual.
    - For manual: Click on the number of corners determined by the dimension you provide. The order matters. If you provided dimensions (5,2) for example, you need to choose a starting corner then move in the direction you choose for the rows, selecting the other 4 corners in the row before moving onto the next row. The next row must start in the same column as the selected starting corner. Below is an example diagram.
        1 > 2 > 3 > 4 > 5
        6 > 7 > 8 > 9 > 10
    - Mistakes: If you make a mistake press 'r' to clear the selected points. You will need to reselect the window again.
- Step 5: Once all points are selected press 'c' to confirm. For intrinic calibration this will now bring up the next image which you will calibrate. For extrinsic calibration or when you are on the final image in the intrinsic calibration, this will complete the program.
"""


import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

class PointSelector:
    def __init__(self, image_folder, img_scaling_factor):
        self.image_folder = image_folder
        self.clicked_points = []
        self.selected_image = None
        self.image_selected = False
        self.img_scaling_factor = 1/img_scaling_factor
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.calType = ''
        self.chessboard_size = (0,0)

        # Initialize the window and set the mouse callback
        cv2.namedWindow('Calibration Image')
        cv2.setMouseCallback('Calibration Image', self.get_click_coordinates)

    def get_click_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Check if left mouse button was clicked
            if not self.image_selected:
                # Capture the current frame as the selected image
                self.selected_image = self.current_frame.copy()
                self.image_selected = True
                print("Image selected for point selection.")
            else:
                if self.calType == 'a':
                    n = 4
                elif self.calType == 'm':
                    n = self.chessboard_size[0]*self.chessboard_size[1]
                
                if self.calType != '':
                    if len(self.clicked_points) < n:
                        self.clicked_points.append((x, y))  # Append the coordinates
                        print(f"Clicked coordinates: ({x}, {y})")  # Print coordinates
                    else:
                        print(f"Already clicked {n} points. Press 'r' to reset. Press 'c' to continue.")

    def draw_points(self, frame):
        for point in self.clicked_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw clicked points
        return frame

    def reset_points(self):
        self.clicked_points.clear()
        print("Reset clicked points.")
        self.image_selected = False

    def run(self, img_id, mode, chessboard_size):
        mode_str = mode.upper()
        running = True
        corners_detected = False
        self.calType = 'a'
        n = 4
        self.chessboard_size = chessboard_size
        while running:
            img_path = os.path.join(self.image_folder, self.image_files[img_id])
            frame = cv2.imread(img_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {img_path}")

            shape = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape[::-1]
            
            frame = cv2.resize(frame, [int(frame.shape[1]*self.img_scaling_factor), int(frame.shape[0]*self.img_scaling_factor)])

            self.current_frame = frame  # Store the current frame for selection
            if not self.image_selected:
                cv2.imshow('Calibration Image', frame)
            else:
                drawed_frame = self.draw_points(self.selected_image)  # Draw points on selected image
                cv2.imshow('Calibration Image', drawed_frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # Reset points if 'r' is pressed, otherwise continue to draw the mask
            if key == ord('r'):
                self.reset_points()
            elif key == ord('s'):
                print("Skipped Image.")
                running = False
                corners = None
                self.reset_points()
            elif key==ord('m'):
                print("Selected Manual Calibration")
                self.calType='m'
                n = self.chessboard_size[0]*self.chessboard_size[1]
            elif key == ord('a'):
                print("Selected Automatic Calibration")
                self.calType = 'a'
                n = 4
            elif key == ord('c'): # wait for 'c' key
                if len(self.clicked_points) < n:
                    print(f"Need to choose {n} points. Press 'r' to reset or continuing choosing point(s) by clicking.")
                else:
                    # Save the selected frame
                    selected_frame = self.selected_image
                    save_img(selected_frame, 'processed_images', f'{mode_str}_SELECTED_{img_id}.jpg')

                    if self.calType == 'a':
                        # Save the masked frame
                        mask = np.zeros((selected_frame.shape[0], selected_frame.shape[1]), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, np.array(self.clicked_points, dtype=np.int32), 255)
                        masked_frame = selected_frame * mask[:, :, np.newaxis]
                        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                        save_img(masked_frame, 'processed_images', f'{mode_str}_MASKED_{img_id}.jpg')

                        # Preprocess the image for better detection
                        # Convert to grayscale
                        gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

                        # Enhance contrast using CLAHE
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)

                        # Apply thresholding
                        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # Save the preprocessed image for debugging
                        cv2.imwrite(f'debug_images/preprocessed_{img_id}.jpg', gray)

                        # Find the chessboard corners with improved flags
                        ret, corners = cv2.findChessboardCorners(
                            gray, chessboard_size,
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                        )

                        # Show corners
                        if ret:
                            running = False
                            corners_detected = True
                            output_image = masked_frame.copy()
                            for idx, corner in enumerate(corners):
                                x, y = int(corner[0][0]), int(corner[0][1])  # Extract x and y coordinates
                                # Draw a red filled circle at each corner
                                cv2.circle(output_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                                # Display the index near each point
                                cv2.putText(output_image, str(idx), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.5, color=(255, 0, 0), thickness=1)  # Blue text with small font

                            # Save the result
                            cv2.imwrite(f"debug_images/debug_points_{img_id}.jpg", output_image)
                        else:
                            print("Failed to find chessboard corners. Repeat the process again.")
                            self.image_selected = False
                            self.reset_points()
                    elif self.calType == 'm':
                        corners = np.array(self.clicked_points, dtype=np.float32).reshape(-1, 1, 2)
                        running = False
                        corners_detected = True
                    
        return corners, shape, corners_detected

def save_calibration_intrinsic(mtx, dist):
    os.makedirs('./calibrations/intrinsic', exist_ok=True)
    # Save intrinsic parameters
    fs = cv2.FileStorage(f'./calibrations/intrinsic/intr.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', mtx)
    fs.write('distortion_coefficients', dist)
    fs.release()

def save_calibration_extrinsic(rvec, tvec):
    os.makedirs('./calibrations/extrinsic', exist_ok=True)
    fs = cv2.FileStorage(f'./calibrations/extrinsic/extr.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('rvec', rvec)
    fs.write('tvec', tvec)  # convert millimeter to meter
    fs.release()

def save_img(img, folder, filename):
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, filename), img)

def calibration(all_frames_corners, img_shape, mode, chessboard_size, square_size):
    # Prepare object points (3D points in real world space)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    for corners in all_frames_corners:
        if corners is not None:
            objpoints.append(objp)  # Add object points
            imgpoints.append(corners)  # Add image points

    # Check if any corners were detected
    if not objpoints or not imgpoints:
        print("Error: No chessboard corners were detected in any images. Calibration cannot proceed.")
        return

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # evaluate error
    if mode == 'intrinsic':
        # Initialize variables for error calculation
        total_error = 0
        total_points = 0

        # Loop through all calibration images
        for i in range(len(objpoints)):
            # Project the 3D points back into the image
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

            # Flatten the arrays for easy calculation
            imgpoints2 = imgpoints2.reshape(-1, 2)
            imgpoints_i = imgpoints[i].reshape(-1, 2)

            # Calculate the error for this image
            error = cv2.norm(imgpoints_i, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            total_points += len(imgpoints2)

        # Calculate the average reprojection error
        mean_error = total_error / total_points
        print(f"Mean Reprojection Error: {mean_error:.4f}")

    # Save the parameters
    if mode == 'intrinsic':
        save_calibration_intrinsic(mtx, dist)
    elif mode == 'extrinsic':
        save_calibration_extrinsic(np.mean(rvecs, axis=0), np.mean(tvecs, axis=0))
    else:
        raise ValueError("Unrecognized calibration mode.")
    
IMAGE_SCALING_FACTOR = 5  # Adjusted to zoom out (scale down to 50% of original size)

# Usage
if __name__ == "__main__":  
    #Set to 'intrinsic' or 'extrinsic' depending on the calibration you want.
    mode = 'intrinsic'
    #mode = 'extrinsic'
    
    #Chessboard dimensions and the square size.
    # Updated to match the 8x8 chessboard (7x7 inner corners)
    chessboard_size = (7, 7)  # 8x8 squares
    square_size = 0.03  # Adjust based on your chessboard's square size (in meters)

    if mode == 'intrinsic':
        # Updated folder path as per the new directive
        intr_image_folder = r'C:\Users\Josh\OneDrive\josh\'s folder\Curtin University\Thesis\Code\Drone_5\wide_cal_frames'
        
        #Number of images in the folder or number of images you want to use.
        num_imgs = 1  # Set to 1 for testing with a single image; adjust as needed
        
        # Check if the folder exists and contains images
        if not os.path.exists(intr_image_folder):
            print(f"Error: The folder '{intr_image_folder}' does not exist. Please check the path.")
            # Fall back to folder selection if the hardcoded path fails
            root = tk.Tk()
            root.withdraw()
            print("Please select the folder containing your chessboard images.")
            intr_image_folder = filedialog.askdirectory(title="Select Folder with Chessboard Images")
            root.destroy()
            if not intr_image_folder:
                print("Error: No folder selected. Exiting.")
                exit()

        image_files = [f for f in os.listdir(intr_image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"Error: No .jpg, .jpeg, or .png images found in '{intr_image_folder}'. Please add images to the folder.")
            exit()
        print(f"Found {len(image_files)} images in '{intr_image_folder}'.")

        all_frames_corners = [None] * num_imgs
        
        #Select corners
        for i in range(min(num_imgs, len(image_files))):
            point_selector = PointSelector(intr_image_folder, IMAGE_SCALING_FACTOR)
            corners, img_shape, corners_detected = point_selector.run(i, mode, chessboard_size)
            all_frames_corners[i] = corners
            if not corners_detected:
                print(f"Warning: No corners detected for image {i}. You can retry with automatic ('a') or manual ('m') mode, or skip with 's'.")
            
        #Calibrate Image      
        calibration(all_frames_corners, img_shape, mode, chessboard_size, square_size)
    elif mode == 'extrinsic':
        #Path to the folder containing the image you want to calibrate with.
        extr_image_folder = "Drone_1/"
        #Path to the intrinsic XML for the camera used to take the image.
        intr_xml_path = "Drone_1/video/intr_1_video_zoom.xml"
        
        #Select corners
        point_selector = PointSelector(extr_image_folder, IMAGE_SC)