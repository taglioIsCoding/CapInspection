import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

BINARY_TRESHOLD = 20
WARP_COEF = 3.5
ANNULAR_REGION_SIZE = 15
# used for better readability while using connected components' stats
WIDTH = 2
HEIGHT = 3
AREA = 4

def cap_preprocessing(path, flag_otsu=False):
    try: 
        filename = path.split("/")[-1]
        image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        
        ## binarization by thresholding the pixel intensity
        if not flag_otsu:
            _, image_bin = cv2.threshold(image, BINARY_TRESHOLD, 255, cv2.THRESH_BINARY)
        else:
            # using Otsu's thresholding for unstable lighting conditions
            _, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        ## Step: 1.1
        center, radius = cap_outline(file=filename, image_bin=image_bin)
        
        radius_inside = int(np.ceil(radius))
        center = (int(center[0]), int(center[1]))
        
        ## Step: 1.2.1
        radius_outside = find_annular_region(radius_inside)
        
        ## Step: 1.2.2
        circular_roi_opened = mask_cap(image_bin, center, radius_inside, radius_outside)

        ## Step: 1.2.3
        tab_center = get_tab_center(file=filename, roi=circular_roi_opened)
        tab_center = np.int32(np.around(tab_center))
        
        ## Step: 1.2.4
        image_with_vertical_tab = rotate_image(center=center, tab_center=tab_center, image=image)

        ## Step: 1.2.5
        cart_corner_coords = extract_corner_coords(radius=radius, center=center)
        
        ## Step: 2
        polar_image, warp_radius = warp_image(radius=radius, image_with_vertical_tab=image_with_vertical_tab, center=center) 
        
        ## Step: 3
        warped_corners = find_warped_corners(corners_coords=cart_corner_coords, center=center, warp_radius=warp_radius, polar_image=polar_image)
        
        # Saving final rectified crop
        rectified_crop = cv2.rotate(polar_image[int(min(warped_corners[1])):int(max(warped_corners[1])),int(min(warped_corners[0])):int(max(warped_corners[0]))], cv2.ROTATE_90_COUNTERCLOCKWISE)
        dir = "crop_imgs/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        name = "result_" + filename
        res_path = dir + name
        
        # Plotting for visualization
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(40, 40)
        ax1.set_title(filename, fontsize=20)
        ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax2.set_title(name, fontsize=20)
        ax2.imshow(rectified_crop, cmap='gray', vmin=0, vmax=255)
        return cv2.imwrite(res_path, rectified_crop)
    
    except Exception as e:
        print(f"An error occurred:\n\t{e}")
        return False
    


##############################################
## Step 1.1: outlining the cap mouth circle ##
##############################################
def cap_outline(file, image_bin):    

    # 2. applying erosion on the binarized image, and then subtracting the result to the original binarized image
    edge_detected_image = image_bin - cv2.erode(image_bin, np.ones((3,3), np.uint8))

    # 3. finding the cap radius by search in the middle vertical line
    middle_line = edge_detected_image[:, int(edge_detected_image.shape[1]/2)]
    non_zero_indices = np.nonzero(middle_line)[0]
    diameter = non_zero_indices[-1]-non_zero_indices[0]
    radius = int(diameter/2)

    rows = image_bin.shape[0]
    found = False
    par2 = 20
    while not found and par2 > 5:
        # 4. applying HoughCircles
        circles = cv2.HoughCircles(edge_detected_image, 
                                cv2.HOUGH_GRADIENT, # Detection method unique that is implemented
                                1, 
                                rows/4,
                                param1=500, # Theshold on the gradient
                                param2=par2,
                                minRadius=int(radius*0.9),
                                maxRadius=int(radius*1.1))  # Smaller find more false positives
        if circles is not None and len(circles) > 0:
            found = not found
        par2 -= 5

    if circles is not None:
        if len(circles[0]) > 1:
            raise ValueError(f"Image {file}: has 2 or more circles")
            
        for circle in circles[0]:
            center = [circle[0], circle[1]]
            radius = circle[2]
    else:
        raise ValueError(f"Image {file}: no circles")
    
    return center, radius

###############################################################
## Step 1.2.1: Finding the annular region containing the tab ##
###############################################################
def find_annular_region(radius):
    radius_outside = radius + ANNULAR_REGION_SIZE
    return radius_outside


############################################################
## Step 1.2.2: Mask the cap to isolate the annular region ##
############################################################
def mask_cap(image_bin, center, radius_inside, radius_outside):
    mask = np.zeros(image_bin.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius_outside, (255), thickness=-1)
    cv2.circle(mask, center, radius_inside, (0), thickness=-1)
    circular_roi = cv2.bitwise_and(image_bin, image_bin, mask=mask)
    circular_roi_open = cv2.morphologyEx(circular_roi, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)
    return circular_roi_open


#################################
## Step 1.2.3: Finding the tab ##
#################################
def get_tab_center(file, roi):
    _, _, stats, centroids = cv2.connectedComponentsWithStats(roi)

    if len(centroids) > 2:
        print(f"{file}: Found {len(centroids) - 1} object in the anular region, they are too many")
        best_index = -1
        best_rect = 0

        for j, stat in enumerate(stats[1:]):
            rect = stat[AREA] / (stat[WIDTH] * stat[HEIGHT])
            if rect > best_rect:
                best_rect = rect
                best_index = j+1 # we start from one to skip the background
        centroids = [centroids[0], centroids[best_index]]

    elif len(centroids) < 2:
        raise ValueError(f"Image {file}: tab not found!")

    # 0 is the background, 1 is the tab
    return centroids[1]

#########################################################
## Step 1.2.4: Finding the angle, and perform rotation ##
#########################################################

def rotate_image(center, tab_center, image):
    # m -> Slope of the straight line that connects the center of the tab with the center of the cap
    m = (center[1] - tab_center[1]) / (center[0] - tab_center[0]) * 1.0

    # Find the angle between the green and the blue lines
    dx = np.abs(tab_center[0]-center[0])
    dy = np.abs(tab_center[1]-center[1])
    teta = np.arctan(dx/dy)

    # Form rad to deg
    rotation = (teta * 180 / np.pi)
    
    # Looking at the slope we choose the direction of the rotation
    if m > 0 :
        rotation = rotation * -1

    # Looking at the tab position we choose if we have to overturn
    if tab_center[1] > center[1]:
        rotation += 180
        
    rot_mat = cv2.getRotationMatrix2D(np.asarray(center, dtype=np.float32), rotation, 1.0)
    
    image_with_vertical_tab = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
    
    return image_with_vertical_tab


#########################################################
## Step 1.2.5: Extracting Cartesian corner coordinates ##
#########################################################
def extract_corner_coords(center, radius):
    top_height = int(center[1]) - int(radius*0.72)
    bottom_height = int(center[1]) - int(radius*0.4)
    left_width = int(center[0]) - int(radius*0.4)
    right_width = int(center[0]) + int(radius*0.4)

    top_left = (top_height, left_width)
    bottom_left = (bottom_height, left_width)
    top_right = (top_height, right_width)
    bottom_right = (bottom_height, right_width)
    
    return [top_left, top_right, bottom_right, bottom_left]

#####################################################################
## Step 2: Applying a Polar Transform to the image with tab on top ##
#####################################################################
def warp_image(radius, image_with_vertical_tab, center):
    flags = cv2.WARP_FILL_OUTLIERS | cv2.WARP_POLAR_LINEAR
    warp_radius = WARP_COEF*radius
    polar_image = cv2.warpPolar(image_with_vertical_tab, image_with_vertical_tab.shape, np.asarray(center, dtype=np.float32), warp_radius, flags)
    return polar_image, warp_radius

###################################################
## Step 3: Extracting Polar corner coordinates ##
###################################################
def find_warped_corners(corners_coords, center, warp_radius, polar_image):
    warped_corners = [[],[]]

    for corner in corners_coords:
        x = corner[1]-center[0]
        y = corner[0]-center[1]
        corner_wrt_center = (x,y)

        angle = 2*np.pi - np.arccos(np.clip(np.dot(corner_wrt_center / np.linalg.norm(corner_wrt_center), (1, 0)), -1.0, 1.0))

        y_warped = (polar_image.shape[0] / (2 * np.pi)) * angle
        x_warped = ((polar_image.shape[1]*1.0) / warp_radius) * np.sqrt(x**2 + y**2)
        
        warped_corners[0].append(x_warped)
        warped_corners[1].append(y_warped)
    
    return warped_corners





