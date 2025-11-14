
"""
Advanced Chessboard Detection System with Manual Corner Selection
Allows manual corner picking on first frame, then automatically matches corners on subsequent frames.
"""
import cv2
import numpy as np
import os
import json
import pickle
from typing import List, Tuple, Optional, Dict
from pathlib import Path
class ManualChessboardDetector:
    """Chessboard detector with manual corner selection and automatic matching."""
    
    def __init__(self, square_size_mm: float = 500.0):
        """
        Initialize the detector.
        
        Args:
            square_size_mm: Real-world size of chessboard squares in millimeters
        """
        self.square_size_mm = square_size_mm
        self.square_size_m = square_size_mm / 1000.0
        self.manual_corners = None
        self.board_size = None
        self.reference_frame = None
        self.template_descriptors = None
        self.corner_templates = []
        
        # Initialize feature detector for template matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Click callback state
        self.clicked_points = []
        self.current_image = None
        self.window_name = "Manual Corner Selection"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for manual corner selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            print(f"Corner {len(self.clicked_points)}: ({x}, {y})")
            
            # Draw the point
            cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.current_image, str(len(self.clicked_points)), 
                       (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(self.window_name, self.current_image)
    
    def manual_corner_selection(self, image: np.ndarray, board_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Allow user to manually select chessboard corners.
        
        Args:
            image: Input image
            board_size: Expected board size (width, height) in internal corners
            
        Returns:
            Array of selected corner coordinates
        """
        self.current_image = image.copy()
        self.clicked_points = []
        
        if board_size is None:
            expected_corners = input("How many corners should be selected? (e.g., 54 for 9x6): ")
            try:
                expected_corners = int(expected_corners)
                # Try to infer board size
                for w in range(3, 15):
                    for h in range(3, 15):
                        if w * h == expected_corners:
                            board_size = (w, h)
                            break
                    if board_size:
                        break
            except:
                expected_corners = 54
                board_size = (9, 6)
        else:
            expected_corners = board_size[0] * board_size[1]
        
        print(f"\nManual Corner Selection Mode")
        print(f"Expected board size: {board_size} ({expected_corners} corners)")
        print("Instructions:")
        print("1. Click on chessboard corners in order (left-to-right, top-to-bottom)")
        print("2. Start from TOP-LEFT corner")
        print("3. Go row by row: first row left-to-right, then second row left-to-right, etc.")
        print("4. Press 'r' to reset if you make a mistake")
        print("5. Press 's' to save when you have all corners")
        print("6. Press 'q' to quit without saving")
        print(f"7. Select exactly {expected_corners} corners")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.current_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                print("Resetting corner selection...")
                self.clicked_points = []
                self.current_image = image.copy()
                cv2.imshow(self.window_name, self.current_image)
                
            elif key == ord('s'):  # Save
                if len(self.clicked_points) == expected_corners:
                    print(f"Saved {len(self.clicked_points)} corners!")
                    break
                else:
                    print(f"Need exactly {expected_corners} corners, but have {len(self.clicked_points)}")
                    
            elif key == ord('q'):  # Quit
                print("Cancelled corner selection")
                self.clicked_points = []
                break
        
        cv2.destroyWindow(self.window_name)
        
        if len(self.clicked_points) == expected_corners:
            corners = np.array(self.clicked_points, dtype=np.float32).reshape(-1, 1, 2)
            self.manual_corners = corners
            self.board_size = board_size
            self.reference_frame = image.copy()
            
            # Create corner templates for matching
            self._create_corner_templates(image, corners)
            
            return corners
        
        return None
    
    def _create_corner_templates(self, image: np.ndarray, corners: np.ndarray):
        """Create templates around each corner for matching."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.corner_templates = []
        template_size = 20  # 20x20 pixel templates
        
        for corner in corners:
            x, y = int(corner[0, 0]), int(corner[0, 1])
            
            # Extract template around corner
            x1, y1 = max(0, x - template_size), max(0, y - template_size)
            x2, y2 = min(gray.shape[1], x + template_size), min(gray.shape[0], y + template_size)
            
            if x2 - x1 >= template_size and y2 - y1 >= template_size:
                template = gray[y1:y2, x1:x2]
                
                # Get ORB features for this template
                kp, desc = self.orb.detectAndCompute(template, None)
                
                self.corner_templates.append({
                    'template': template,
                    'center': (x - x1, y - y1),  # Center relative to template
                    'keypoints': kp,
                    'descriptors': desc,
                    'original_pos': (x, y)
                })
    
    def automatic_corner_detection(self, image: np.ndarray, 
                                 search_radius: int = 50) -> Optional[np.ndarray]:
        """
        Automatically detect corners based on manual selection from reference frame.
        
        Args:
            image: Input image
            search_radius: Search radius around expected positions
            
        Returns:
            Detected corner coordinates or None if detection fails
        """
        if self.manual_corners is None or self.board_size is None:
            print("No manual corners available. Please run manual selection first.")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        detected_corners = []
        
        print("Attempting automatic corner detection...")
        
        # Method 1: Template matching
        corners_template = self._template_matching_detection(gray, search_radius)
        if corners_template is not None and len(corners_template) == len(self.manual_corners):
            print("Template matching successful!")
            return corners_template
        
        # Method 2: Feature matching
        corners_features = self._feature_matching_detection(gray, search_radius)
        if corners_features is not None and len(corners_features) == len(self.manual_corners):
            print("Feature matching successful!")
            return corners_features
        
        # Method 3: Try OpenCV's built-in detection with known size
        corners_opencv = self._opencv_detection_with_size(gray)
        if corners_opencv is not None and len(corners_opencv) == len(self.manual_corners):
            print("OpenCV detection successful!")
            return corners_opencv
        
        # Method 4: Optical flow tracking (if we have previous frame)
        corners_flow = self._optical_flow_detection(gray)
        if corners_flow is not None and len(corners_flow) == len(self.manual_corners):
            print("Optical flow tracking successful!")
            return corners_flow
        
        print("All automatic detection methods failed.")
        return None
    
    def _template_matching_detection(self, image: np.ndarray, search_radius: int) -> Optional[np.ndarray]:
        """Detect corners using template matching."""
        if not self.corner_templates:
            return None
        
        detected_corners = []
        
        for i, template_info in enumerate(self.corner_templates):
            template = template_info['template']
            expected_x, expected_y = template_info['original_pos']
            
            # Define search region
            x1 = max(0, expected_x - search_radius)
            y1 = max(0, expected_y - search_radius)
            x2 = min(image.shape[1], expected_x + search_radius)
            y2 = min(image.shape[0], expected_y + search_radius)
            
            search_region = image[y1:y2, x1:x2]
            
            if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
                continue
            
            # Template matching
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.6:  # Threshold for good match
                # Convert to global coordinates
                center_x = x1 + max_loc[0] + template_info['center'][0]
                center_y = y1 + max_loc[1] + template_info['center'][1]
                detected_corners.append([center_x, center_y])
        
        if len(detected_corners) == len(self.corner_templates):
            return np.array(detected_corners, dtype=np.float32).reshape(-1, 1, 2)
        
        return None
    
    def _feature_matching_detection(self, image: np.ndarray, search_radius: int) -> Optional[np.ndarray]:
        """Detect corners using feature matching."""
        if not self.corner_templates:
            return None
        
        # Get features from current image
        kp_current, desc_current = self.orb.detectAndCompute(image, None)
        if desc_current is None:
            return None
        
        detected_corners = []
        
        for template_info in self.corner_templates:
            if template_info['descriptors'] is None:
                continue
            
            # Match features
            matches = self.matcher.match(template_info['descriptors'], desc_current)
            
            if len(matches) < 3:
                continue
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:min(10, len(matches))]
            
            # Get matched keypoints
            src_pts = np.float32([template_info['keypoints'][m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches])
            
            # Find transformation
            if len(src_pts) >= 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    # Transform corner position
                    corner_in_template = np.array([[template_info['center']]], dtype=np.float32)
                    transformed_corner = cv2.perspectiveTransform(corner_in_template, M)
                    detected_corners.append(transformed_corner[0, 0])
        
        if len(detected_corners) == len(self.corner_templates):
            return np.array(detected_corners, dtype=np.float32).reshape(-1, 1, 2)
        
        return None
    
    def _opencv_detection_with_size(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Try OpenCV detection with known board size."""
        if self.board_size is None:
            return None
        
        # Try different flags
        flags = [
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        ]
        
        for flag in flags:
            try:
                if flag is None:
                    ret, corners = cv2.findChessboardCorners(image, self.board_size, None)
                else:
                    ret, corners = cv2.findChessboardCorners(image, self.board_size, None, flag)
                
                if ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
                    return corners_refined
            except:
                continue
        
        return None
    
    def _optical_flow_detection(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Use optical flow to track corners from previous frame."""
        # This would require storing previous frame - simplified implementation
        return None
    
    def save_manual_calibration(self, filepath: str):
        """Save manual calibration data."""
        data = {
            'manual_corners': self.manual_corners.tolist() if self.manual_corners is not None else None,
            'board_size': self.board_size,
            'square_size_mm': self.square_size_mm,
            'corner_templates': []  # Templates are too complex for JSON, save separately
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save templates separately
        if self.corner_templates:
            template_path = filepath.replace('.json', '_templates.pkl')
            with open(template_path, 'wb') as f:
                pickle.dump(self.corner_templates, f)
        
        print(f"Manual calibration saved to {filepath}")
    
    def load_manual_calibration(self, filepath: str) -> bool:
        """Load manual calibration data."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if data['manual_corners']:
                self.manual_corners = np.array(data['manual_corners'], dtype=np.float32)
                self.board_size = tuple(data['board_size'])
                self.square_size_mm = data['square_size_mm']
                
                # Load templates
                template_path = filepath.replace('.json', '_templates.pkl')
                if os.path.exists(template_path):
                    with open(template_path, 'rb') as f:
                        self.corner_templates = pickle.load(f)
                
                print(f"Manual calibration loaded from {filepath}")
                return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
        
        return False
    
    def create_object_points(self, board_size: Tuple[int, int] = None, 
                           square_size: float = None) -> np.ndarray:
        """Create 3D object points for the chessboard."""
        if board_size is None:
            board_size = self.board_size
        if square_size is None:
            square_size = self.square_size_m
        
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        return objp
    
    def visualize_detection(self, image: np.ndarray, corners: np.ndarray, 
                          title: str = "Detection Result", save_path: str = None) -> np.ndarray:
        """Visualize corner detection results."""
        vis_image = image.copy()
        
        if corners is not None and len(corners) > 0:
            # Draw corners
            corners_2d = corners.reshape(-1, 2)
            for i, (x, y) in enumerate(corners_2d):
                cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(vis_image, str(i), (int(x) + 8, int(y) + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw grid lines if we have the right number of corners
            if self.board_size and len(corners_2d) == self.board_size[0] * self.board_size[1]:
                corners_grid = corners_2d.reshape(self.board_size[1], self.board_size[0], 2)
                
                # Draw horizontal lines
                for i in range(self.board_size[1]):
                    for j in range(self.board_size[0] - 1):
                        pt1 = tuple(corners_grid[i, j].astype(int))
                        pt2 = tuple(corners_grid[i, j + 1].astype(int))
                        cv2.line(vis_image, pt1, pt2, (255, 0, 0), 1)
                
                # Draw vertical lines
                for i in range(self.board_size[1] - 1):
                    for j in range(self.board_size[0]):
                        pt1 = tuple(corners_grid[i, j].astype(int))
                        pt2 = tuple(corners_grid[i + 1, j].astype(int))
                        cv2.line(vis_image, pt1, pt2, (255, 0, 0), 1)
            
            # Add info
            info_text = f"Corners: {len(corners_2d)}"
            if self.board_size:
                info_text += f" | Board: {self.board_size[0]}x{self.board_size[1]}"
            
            cv2.putText(vis_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_image, "No corners detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def analyze_video_frames(self, video_path: str, frame_numbers: List[int],
                           output_dir: str = "manual_chessboard_analysis",
                           calibration_file: str = None) -> Dict:
        """
        Analyze video frames with manual corner selection for ALL target frames,
        then automatic detection for verification and additional frames.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Try to load existing calibration
        calibration_loaded = False
        existing_manual_frames = []
        
        if calibration_file and os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r') as f:
                    existing_data = json.load(f)
                if 'manual_frames' in existing_data:
                    existing_manual_frames = existing_data['manual_frames']
                    print(f"Found existing manual calibration for {len(existing_manual_frames)} frames")
                    calibration_loaded = True
            except Exception as e:
                print(f"Error loading existing calibration: {e}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        results = {
            'video_path': video_path,
            'manual_calibration_frames': [],
            'automatic_detection_frames': [],
            'board_size': None,
            'calibration_file': calibration_file,
            'summary': {
                'manual_frames_completed': 0,
                'automatic_detections_successful': 0,
                'total_target_frames': len(frame_numbers)
            }
        }
        
        # Phase 1: Manual calibration for each target frame
        print(f"\n{'='*60}")
        print("PHASE 1: MANUAL CALIBRATION FOR TARGET FRAMES")
        print(f"{'='*60}")
        
        manual_calibration_data = {'manual_frames': []}
        
        for frame_idx, frame_num in enumerate(frame_numbers):
            print(f"\nManual calibration for frame {frame_num} ({frame_idx + 1}/{len(frame_numbers)})")
            
            # Check if this frame already has manual calibration
            existing_frame = None
            for existing in existing_manual_frames:
                if existing['frame_number'] == frame_num:
                    existing_frame = existing
                    break
            
            if existing_frame:
                print(f"Frame {frame_num} already has manual calibration. Options:")
                print("1. Use existing calibration")
                print("2. Redo manual calibration")
                choice = input("Enter choice (1 or 2): ").strip()
                
                if choice == "1":
                    print("Using existing calibration")
                    manual_calibration_data['manual_frames'].append(existing_frame)
                    
                    # Update internal state for first frame
                    if frame_idx == 0:
                        self.manual_corners = np.array(existing_frame['corners'], dtype=np.float32)
                        self.board_size = tuple(existing_frame['board_size'])
                    
                    continue
            
            # Read the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Could not read frame {frame_num}")
                continue
            
            print(f"Starting manual corner selection for frame {frame_num}...")
            print("Follow the instructions in the popup window")
            
            # Manual corner selection
            corners = self.manual_corner_selection(frame, self.board_size)
            
            if corners is None:
                print(f"Manual selection cancelled for frame {frame_num}")
                choice = input("Continue with next frame? (y/n): ").strip().lower()
                if choice != 'y':
                    break
                continue
            
            # Store manual calibration data
            frame_manual_data = {
                'frame_number': frame_num,
                'corners': corners.tolist(),
                'board_size': self.board_size,
                'corner_count': len(corners)
            }
            
            manual_calibration_data['manual_frames'].append(frame_manual_data)
            
            # Create visualization of manual selection
            vis_image = self.visualize_detection(frame, corners, 
                                               f"Frame {frame_num} - Manual Selection")
            vis_path = output_path / f"frame_{frame_num:06d}_manual.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            
            # Store in results
            results['manual_calibration_frames'].append({
                'frame_number': frame_num,
                'corners_selected': len(corners),
                'board_size': self.board_size,
                'success': True
            })
            
            print(f"Manual calibration completed for frame {frame_num}")
            print(f"Selected {len(corners)} corners")
            
            # Update board size if this is the first successful calibration
            if results['board_size'] is None:
                results['board_size'] = self.board_size
        
        # Save manual calibration data
        if calibration_file and manual_calibration_data['manual_frames']:
            # Load existing data and merge
            final_calibration_data = {'manual_frames': []}
            
            if calibration_loaded:
                # Merge with existing data
                existing_frame_nums = [f['frame_number'] for f in existing_manual_frames]
                for existing in existing_manual_frames:
                    if existing['frame_number'] not in [f['frame_number'] for f in manual_calibration_data['manual_frames']]:
                        final_calibration_data['manual_frames'].append(existing)
            
            # Add new manual calibrations
            final_calibration_data['manual_frames'].extend(manual_calibration_data['manual_frames'])
            
            # Save
            with open(calibration_file, 'w') as f:
                json.dump(final_calibration_data, f, indent=2)
            
            print(f"\nManual calibration data saved to {calibration_file}")
        
        results['summary']['manual_frames_completed'] = len(results['manual_calibration_frames'])
        
        # Phase 2: Automatic detection verification
        print(f"\n{'='*60}")
        print("PHASE 2: AUTOMATIC DETECTION VERIFICATION")
        print(f"{'='*60}")
        
        if not manual_calibration_data['manual_frames']:
            print("No manual calibrations available. Skipping automatic detection.")
            cap.release()
            return results
        
        # Use the first manual calibration as reference for automatic detection
        reference_frame_data = manual_calibration_data['manual_frames'][0]
        self.manual_corners = np.array(reference_frame_data['corners'], dtype=np.float32)
        self.board_size = tuple(reference_frame_data['board_size'])
        
        # Get reference frame for template creation
        ref_frame_num = reference_frame_data['frame_number']
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_num)
        ret, ref_frame = cap.read()
        if ret:
            self.reference_frame = ref_frame.copy()
            self._create_corner_templates(ref_frame, self.manual_corners)
            print(f"Using frame {ref_frame_num} as reference for automatic detection")
        
        # Store images for final review
        review_images = []
        
        # Run automatic detection on all target frames
        for frame_data in manual_calibration_data['manual_frames']:
            frame_num = frame_data['frame_number']
            print(f"\nAutomatic detection verification for frame {frame_num}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Could not read frame {frame_num}")
                continue
            
            # Run automatic detection
            auto_corners = self.automatic_corner_detection(frame)
            method_used = "automatic" if auto_corners is not None else "failed"
            
            # Compare with manual corners if available
            manual_corners = np.array(frame_data['corners'], dtype=np.float32)
            comparison_error = None
            
            if auto_corners is not None:
                # Calculate difference between manual and automatic
                auto_corners_2d = auto_corners.reshape(-1, 2)
                manual_corners_2d = manual_corners.reshape(-1, 2)
                
                if len(auto_corners_2d) == len(manual_corners_2d):
                    differences = np.linalg.norm(auto_corners_2d - manual_corners_2d, axis=1)
                    comparison_error = np.mean(differences)
                    print(f"Average difference from manual: {comparison_error:.2f} pixels")
                else:
                    print(f"Corner count mismatch: auto={len(auto_corners_2d)}, manual={len(manual_corners_2d)}")
            
            # Create comparison visualization
            vis_image = frame.copy()
            
            # Draw manual corners in green
            manual_corners_2d = manual_corners.reshape(-1, 2)
            for i, (x, y) in enumerate(manual_corners_2d):
                cv2.circle(vis_image, (int(x), int(y)), 7, (0, 255, 0), 2)
                cv2.putText(vis_image, f"M{i}", (int(x) + 10, int(y) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw automatic corners in red
            if auto_corners is not None:
                auto_corners_2d = auto_corners.reshape(-1, 2)
                for i, (x, y) in enumerate(auto_corners_2d):
                    cv2.circle(vis_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.putText(vis_image, f"A{i}", (int(x) - 15, int(y) - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw lines connecting corresponding points
                if len(auto_corners_2d) == len(manual_corners_2d):
                    for (mx, my), (ax, ay) in zip(manual_corners_2d, auto_corners_2d):
                        cv2.line(vis_image, (int(mx), int(my)), (int(ax), int(ay)), (255, 0, 0), 1)
            
            # Add comparison info
            info_text = f"Frame {frame_num}: Manual(Green) vs Auto(Red)"
            cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if comparison_error is not None:
                error_text = f"Avg Error: {comparison_error:.1f}px"
                cv2.putText(vis_image, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                error_text = "AUTO DETECTION FAILED"
                cv2.putText(vis_image, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save comparison visualization
            vis_path = output_path / f"frame_{frame_num:06d}_comparison.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            
            # Store for final review
            review_images.append({
                'frame_number': frame_num,
                'image': vis_image,
                'success': auto_corners is not None,
                'error': comparison_error
            })
            
            # Store results
            auto_result = {
                'frame_number': frame_num,
                'automatic_detection_successful': auto_corners is not None,
                'corner_count_auto': len(auto_corners) if auto_corners is not None else 0,
                'corner_count_manual': len(manual_corners_2d),
                'average_error_pixels': comparison_error,
                'method_used': method_used
            }
            
            results['automatic_detection_frames'].append(auto_result)
            
            if auto_corners is not None:
                results['summary']['automatic_detections_successful'] += 1
            
            print(f"Frame {frame_num}: {method_used} - "
                  f"{'Success' if auto_corners is not None else 'Failed'}")
        
        cap.release()
        
        # Phase 3: Interactive review of automatic detection results
        print(f"\n{'='*60}")
        print("PHASE 3: INTERACTIVE REVIEW OF AUTOMATIC DETECTION")
        print(f"{'='*60}")
        
        if review_images:
            print("Showing automatic detection results for all frames...")
            print("Instructions:")
            print("- Press any key to go to next frame")
            print("- Press 'q' to quit review")
            print("- Green circles = Manual corners (ground truth)")
            print("- Red circles = Automatic detection results")
            print("- Blue lines = Correspondence between manual and automatic")
            
            cv2.namedWindow("Automatic Detection Review", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Automatic Detection Review", 1200, 800)
            
            for i, img_data in enumerate(review_images):
                frame_num = img_data['frame_number']
                image = img_data['image']
                success = img_data['success']
                error = img_data['error']
                
                # Add frame counter
                counter_text = f"Frame {i+1}/{len(review_images)}"
                cv2.putText(image, counter_text, (10, image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the image
                cv2.imshow("Automatic Detection Review", image)
                
                print(f"\nShowing frame {frame_num} ({i+1}/{len(review_images)})")
                if success:
                    print(f"  Status: SUCCESS - Average error: {error:.2f} pixels")
                else:
                    print(f"  Status: FAILED - No automatic detection")
                
                print("  Press any key to continue, 'q' to quit review...")
                
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("Review stopped by user")
                    break
            
            cv2.destroyWindow("Automatic Detection Review")
            print("\nAutomatic detection review completed!")
        
        # Final summary display
        print(f"\n{'='*60}")
        print("FINAL DETECTION SUMMARY")
        print(f"{'='*60}")
        
        for img_data in review_images:
            frame_num = img_data['frame_number']
            success = img_data['success']
            error = img_data['error']
            
            status = "SUCCESS" if success else "FAILED"
            error_str = f"({error:.2f}px)" if error is not None else ""
            
            print(f"Frame {frame_num}: {status} {error_str}")
        
        successful_count = sum(1 for img in review_images if img['success'])
        print(f"\nOverall: {successful_count}/{len(review_images)} frames successful")
        
        # Save complete results
        results_path = output_path / "complete_calibration_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comprehensive summary
        self._print_comprehensive_summary(results)
        
        return results
    
    def _print_comprehensive_summary(self, results: Dict):
        """Print comprehensive analysis summary."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE CALIBRATION SUMMARY")
        print(f"{'='*70}")
        
        # Manual calibration summary
        manual_frames = results['summary']['manual_frames_completed']
        total_frames = results['summary']['total_target_frames']
        print(f"Manual Calibration:")
        print(f"  Completed: {manual_frames}/{total_frames} frames")
        print(f"  Success rate: {manual_frames/total_frames*100:.1f}%")
        
        if results['board_size']:
            print(f"  Board size detected: {results['board_size'][0]}x{results['board_size'][1]}")
        
        # Automatic detection summary
        auto_successful = results['summary']['automatic_detections_successful']
        auto_total = len(results['automatic_detection_frames'])
        
        if auto_total > 0:
            print(f"\nAutomatic Detection Verification:")
            print(f"  Successful: {auto_successful}/{auto_total} frames")
            print(f"  Success rate: {auto_successful/auto_total*100:.1f}%")
            
            # Error statistics
            errors = [r['average_error_pixels'] for r in results['automatic_detection_frames'] 
                     if r['average_error_pixels'] is not None]
            if errors:
                print(f"  Average error: {np.mean(errors):.2f} pixels")
                print(f"  Max error: {np.max(errors):.2f} pixels")
                print(f"  Min error: {np.min(errors):.2f} pixels")
        
        print(f"\nFiles Generated:")
        print(f"  Manual calibration images: frame_XXXXXX_manual.jpg")
        print(f"  Comparison images: frame_XXXXXX_comparison.jpg")
        print(f"  Results file: complete_calibration_results.json")
        print(f"  Calibration data: {results.get('calibration_file', 'Not specified')}")
        
        print(f"{'='*70}")

def main():
    """Example usage."""
    # Your video path
    video_path = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\DJI-Mavic3E-Drone4-20250509T044838Z-001\DJI-Mavic3E-Drone4\DJI_202501171525_017\DJI_20250117153008_0002_V.MP4"
    
    # Create detector
    detector = ManualChessboardDetector(square_size_mm=500.0)
    
    # Target frames
    target_frames = [201, 1382, 500, 414, 380]
    
    # Calibration file to save/load manual selection
    calibration_file = "manual_chessboard_calibration.json"
    
    if os.path.exists(video_path):
        try:
            results = detector.analyze_video_frames(
                video_path=video_path,
                frame_numbers=target_frames,
                output_dir="manual_chessboard_results",
                calibration_file=calibration_file
            )
            
            print("\nSummary:")
            successful = sum(1 for r in results['frame_results'] if r['corners_detected'])
            print(f"Successfully detected corners in {successful}/{len(target_frames)} frames")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Video file not found: {video_path}")
if __name__ == "__main__":
    main()
