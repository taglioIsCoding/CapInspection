import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def cap_preprocessing(path):
    try: 
        filename = path.split("/")[-1]
        image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        image = image[:,:, 0]
        
        ## Binarization by thresholding the pixel intensity
        _, image_bin = cv2.threshold(image.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
        
        ## Step: 1.1
        center, radius = cap_outline(file=filename, image_bin=image_bin)
        
        radius_inside = int(np.ceil(radius))
        center = (int(np.ceil(center[0])), int(np.ceil(center[1])))
        
        ## Step: 1.2.1
        radius_outside = find_annular_region(radius_inside)
        
        ## Step: 1.2.2
        circular_roi_opened = mask_cap(image_bin, center, radius_inside, radius_outside)

        ## Step: 1.2.3
        tab_center = get_tab_center(file=filename, roi=circular_roi_opened)
        tab_center = np.int32(np.around(tab_center))
        center = np.int32(np.around(center))
        
        ## Step: 1.2.4
        image_with_vertical_tab = rotate_image(center=center, tab_center=tab_center, image=image)

        ## Step: 1.2.5
        cart_corner_coords = extract_corner_coords(radius=radius, center=center)
        
        ## Step: 2
        flags = cv2.INTER_CUBIC | cv2.WARP_FILL_OUTLIERS | cv2.WARP_POLAR_LINEAR
        warp_radius = 3.5*radius
        polar_image = cv2.warpPolar(image_with_vertical_tab, image_with_vertical_tab.shape, np.asarray(center, dtype=np.float32), warp_radius, flags)
        warped_corners = find_warped_corners(corners_coords=cart_corner_coords, center=center, warp_radius=warp_radius, polar_image=polar_image)
        
        # Saving final rectified crop
        rectified_crop = cv2.rotate(polar_image[int(min(warped_corners[1])):int(max(warped_corners[1])),int(min(warped_corners[0])):int(max(warped_corners[0]))], cv2.ROTATE_90_COUNTERCLOCKWISE)
        name = "result_" + filename + ".jpg"
        
        return cv2.imwrite(name, rectified_crop)
    
    except Exception as e:
        print(f"An error occurred:\n\t{e}")
        return False
    


##############################################
## Step 1.1: outlining the cap mouth circle ##
##############################################
def cap_outline(file, image_bin):    

    # 2. applying erosion on the binarized image, and then subtracting the result to the original binarized image
    edge_detected_image = image_bin - cv2.morphologyEx(image_bin, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=1)

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
            print(f"Image {file} has 2 or more circles")
            
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
    radius_outside = radius + 15
    return radius_outside


############################################################
## Step 1.2.2: Mask the cap to isolate the annular region ##
############################################################
def mask_cap(image_bin, center, radius_inside, radius_outside):
    mask = np.zeros(image_bin.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius_outside, (255), thickness=-1)
    cv2.circle(mask, center, radius_inside, (0), thickness=-1)
    circular_roi = cv2.bitwise_and(image_bin, image_bin, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    circular_roi_open = cv2.morphologyEx(circular_roi, cv2.MORPH_OPEN, kernel, iterations=2)
    return circular_roi_open


#################################
## Step 1.2.3: Finding the tab ##
#################################
def get_tab_center(file, roi):
    _, _, stats, centroids = cv2.connectedComponentsWithStats(roi)

    if len(centroids) > 2:
        print(f"{file}: Found {len(centroids) - 1} object in the anular region, they are too many")
        WIDTH = 2
        HEIGHT = 3
        AREA = 4
        best_index = -1
        best_rect = 0
        # The residual object are the tab and some arc region that is too big to be removed with the opening
        # We can find the tab looking at the most rectangular object 
        for j, stat in enumerate(stats[1:]): # Altrenativa a median: [x for x in stats[1:] if x[AREA] > 20]
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
    x = np.abs(tab_center[0]-center[0])
    y = np.abs(tab_center[1]-center[1])
    teta = np.arctan(x/y)

    # Form rad to deg
    rotation = (teta * 180 / np.pi)
    
    # Looking at the slope we choose the direction of the rotation
    if m > 0 :
        rotation = rotation * -1

    # Looking at the tab position we choose if we have to overturn
    if tab_center[1] > center[1]:
        rotation += 180
        
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
    
    image_with_vertical_tab = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
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

###################################################
## Step 2.1: Extracting Polar corner coordinates ##
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





