import cv2
import projection_utils as pu
import os

# Specify the path to exiftool.exe (corrected filename)
exiftool_path = r"C:\Program Files\exiftool\exiftool-13.30_64\exiftool(-k).exe"

# Verify ExifTool executable exists
if not os.path.exists(exiftool_path):
    print(f"Error: ExifTool executable not found at {exiftool_path}")
    exit()

# Image used for the initial extrinsic calibration.
image_path0 = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\Code\Joshua_Honour_Projects\camera Calibration\Drone_2\photo\extrinsic\wide\test0.JPG"
# Path to the initial extrinsic XML file.
init_extr_xml_path = r"camera Calibration\Drone_4\video\extrinsic\extr_0.xml"

# Image you want to project onto.
image_path1 = r"C:\Users\euroc\OneDrive\josh's folder\Curtin University\Thesis\Code\Joshua_Honour_Projects\camera Calibration\Drone_2\photo\extrinsic\wide\test2.JPG"
# Path to the intrinsic XML file for the camera used to take image 1.
intr_xml_path = r"camera Calibration\Drone_4\video\intrinsic\intr_4_video_wide.xml"

# Chessboard square size used during both intrinsic and extrinsic calibrations.
sq_size = 0.5

# Chessboard dimensions (number of inner corners: width, height)
dim = (5, 4)  # Reduced from (9, 6) to decrease point density; adjust as needed

# Retrieve metadata with exiftool_path
gps0, angles0 = pu.getImageMetadata(image_path0, exiftool_path=exiftool_path)
gps1, angles1 = pu.getImageMetadata(image_path1, exiftool_path=exiftool_path)

t_edn = pu.getEDNTranslation(gps0, gps1)

intr_mat, dist_coeffs = pu.getIntrinsicData(intr_xml_path)
init_extr = pu.getInitialExtrinsicMatrix(init_extr_xml_path)
rel_extr = pu.getRelativeExtrinsicMatrix(angles0, angles1, t_edn)

image = cv2.imread(image_path1)
# Updated call to include the dim parameter
image, points = pu.projectPoints(image, dim, sq_size, intr_mat, dist_coeffs, init_extr, rel_extr)

# Debug: Print the number of projected points and a sample
print(f"Number of projected points: {len(points)}")
if points:
    print(f"Sample point: {points[0]}")

# Resize the image for display (adjust scale_factor as needed)
scale_factor = 0.5  # Reduces image size by 50%; increase to 1.0 for full size
display_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

# Create a resizable window
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Drone 0: {gps0}, {angles0}')
print(f'Drone 1: {gps1}, {angles1}')
print(f'EDN Translation: {t_edn}')