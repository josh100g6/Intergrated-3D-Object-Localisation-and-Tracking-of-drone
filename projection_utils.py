import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
import exiftool
import re
import os  # Added this import to fix the NameError

def getIntrinsicData(xml_path):
    intr_xml = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    intr_mat = intr_xml.getNode("camera_matrix").mat()
    dist_coeffs = intr_xml.getNode("distortion_coefficients").mat()
    return intr_mat, dist_coeffs

def getInitialExtrinsicMatrix(xml_path):
    extr_xml = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    rvec = extr_xml.getNode("rvec").mat()
    tvec = extr_xml.getNode("tvec").mat()
    rmat, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmat, tvec))

def getRelativeExtrinsicMatrix(angles0, angles1, t_edn):
    # Rotates from Camera 0's frame to the EDN frame. Eq 71.
    R_0_edn = R.from_euler('zxy', [angles0[2], angles0[1], angles0[0]], degrees=True).as_matrix()
    # Rotates from the EDN frame to Camera 1's frame. Eq 72.
    R_edn_1 = R.from_euler('yxz', [-angles1[0], -angles1[1], -angles1[2]], degrees=True).as_matrix()
    # Converts the translation from the EDN frame to Camera 1's frame. Eq 74.
    t_1 = R_edn_1 @ t_edn
    # Rotates directly from Camera 0's frame to Camera 1's frame. Eq 73.
    R_0_1 = R_edn_1 @ R_0_edn
    return np.hstack((R_0_1, t_1))

def getEDNTranslation(gps0, gps1):
    n, e, d = pm.geodetic2ned(*gps0, *gps1)
    return [[float(e)], [float(d)], [float(n)]]

def projectPoints(image, dim, sq_size, intr_mat, dist_coeffs, init_extr, rel_extr):
    n, m = dim
    Pws = np.zeros((n*m, 3), np.float64)
    Pws[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2) * sq_size   
    
    points = []
    for Pw in Pws:
        # Converts from the coordinate system produced during the initial extrinsic calibration to Camera 0's coordinate system. Eq 68.
        Pc0 = init_extr @ np.append(Pw, 1)
    
        # Converts from Camera 0's coordinate system to Camera 1's. Eq 75.
        Pc1 = rel_extr @ np.append(Pc0, 1)
        
        # Projects 3D coordinates in Camera 1's coordinate system to Camera 1's 2D image plane.
        zero_vec = np.zeros((3, 1))
        p, _ = cv2.projectPoints(Pc1, zero_vec, zero_vec, intr_mat, dist_coeffs)
        x = int(p[0][0][0])
        y = int(p[0][0][1])
        points.append([x,y])
        
        colour = (0, 0, 255)
        try:
            cv2.circle(image, (x,y), radius=5, color=colour, thickness=-1)
            cv2.putText(image, str(Pw), (x-50, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                                        fontScale=0.5, color=colour, thickness=1)
        except:
            pass
        
    return image, points

def getImageMetadata(image_path, exiftool_path=None):
    if exiftool_path and os.path.exists(exiftool_path):
        et = exiftool.ExifToolHelper(executable=exiftool_path)
    else:
        et = exiftool.ExifToolHelper()  # Fallback to default PATH
    tags = ['XMP:GPSLatitude', 'XMP:GPSLongitude', 'XMP:RelativeAltitude', 'XMP:GimbalYawDegree', 'XMP:GimbalPitchDegree', 'XMP:GimbalRollDegree']

    metadata = []
    for tag in tags:
        metadata.append(float(et.get_tags(image_path, tag)[0][tag]))
        
    return metadata[:3], metadata[3:]

def getVideoFrameMetadata(srt_file, frame_number):
    with open(srt_file, 'r') as file:
        content = file.read()

    pattern = rf"FrameCnt:\s*{frame_number}.*?\[latitude:\s*([\-0-9\.]+)\]\s*\[longitude:\s*([\-0-9\.]+)\].*?\[rel_alt:\s*([\-0-9\.]+)\s*abs_alt:\s*([\-0-9\.]+)\].*?\[gb_yaw:\s*([\-0-9\.]+)\s*gb_pitch:\s*([\-0-9\.]+)\s*gb_roll:\s*([\-0-9\.]+)\]"
    
    match = re.search(pattern, content, re.DOTALL)

    if match:
        lat, lon, rel_alt, abs_alt, yaw, pitch, roll = match.groups()
        gps = [float(lat), float(lon), float(rel_alt)]
        angles = [float(yaw), float(pitch), float(roll)]
        return gps, angles
    else:
        return f"No data found for frame {frame_number}."