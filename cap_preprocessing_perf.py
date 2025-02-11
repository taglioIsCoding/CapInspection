import cv2
import numpy as np
import sys
import timeit

WARP_COEF = 3.5

if(len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} filename")
    sys.exit(2)

filename = sys.argv[1]

try:
    image = cv2.imread(filename, cv2.COLOR_BGR2GRAY)

    start_time = timeit.default_timer()
    # Binarization by thresholding the pixel intensity
    _, image_bin = cv2.threshold(image.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

    ##############################################
    ## Step 1.1: outlining the cap mouth circle ##
    ##############################################
    
    # applying erosion on the binarized image, and then subtracting the result to the original binarized image
    edge_detected_image = image_bin - cv2.morphologyEx(image_bin, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=1)

    # finding the cap radius by search in the middle vertical line
    middle_line = edge_detected_image[:, int(edge_detected_image.shape[1]/2)]
    non_zero_indices = np.nonzero(middle_line)[0]
    diameter = non_zero_indices[-1]-non_zero_indices[0]
    radius = int(diameter/2)

    rows = image_bin.shape[0]
    found = False
    par2 = 20
    while not found and par2 > 5:
    # applying HoughCircles
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
            print(f"Image {filename} has 2 or more circles")
            
        for circle in circles[0]:
            center = [circle[0], circle[1]]
            radius = circle[2]
    else:
        raise ValueError(f"Image {filename}: no circles")
    
    radius_inside = int(np.ceil(radius))
    center = (int(np.ceil(center[0])), int(np.ceil(center[1])))

    ###############################################################
    ## Step 1.2.1: Finding the annular region containing the tab ##
    ###############################################################
    radius_outside = radius_inside + 15

    ############################################################
    ## Step 1.2.2: Mask the cap to isolate the annular region ##
    ############################################################
    mask = np.zeros(image_bin.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius_outside, (255), thickness=-1)
    cv2.circle(mask, center, radius_inside, (0), thickness=-1)
    circular_roi = cv2.bitwise_and(image_bin, image_bin, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    circular_roi_opened = cv2.morphologyEx(circular_roi, cv2.MORPH_OPEN, kernel, iterations=2)

    #################################
    ## Step 1.2.3: Finding the tab ##
    #################################
    _, _, stats, centroids = cv2.connectedComponentsWithStats(circular_roi_opened)

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
    tab_center = np.int32(centroids[1])
    center = np.int32(np.around(center))

    #########################################################
    ## Step 1.2.4: Finding the angle, and perform rotation ##
    #########################################################
    
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

    #########################################################
    ## Step 1.2.5: Extracting Cartesian corner coordinates ##
    #########################################################
    top_height = int(center[1]) - int(radius*0.72)
    bottom_height = int(center[1]) - int(radius*0.4)
    left_width = int(center[0]) - int(radius*0.4)
    right_width = int(center[0]) + int(radius*0.4)

    top_left = (top_height, left_width)
    bottom_left = (bottom_height, left_width)
    top_right = (top_height, right_width)
    bottom_right = (bottom_height, right_width)
    
    corners_coords =  [top_left, top_right, bottom_right, bottom_left] # cartesian corner coords

    ########################################
    ## Step 2: Producing a rectified crop ##
    ########################################

    flags = cv2.INTER_CUBIC | cv2.WARP_FILL_OUTLIERS | cv2.WARP_POLAR_LINEAR
    warp_radius = WARP_COEF*radius
    polar_image = cv2.warpPolar(image_with_vertical_tab, image_with_vertical_tab.shape, np.asarray(center, dtype=np.float32), warp_radius, flags)

    ###################################################
    ## Step 2.1: Extracting Polar corner coordinates ##
    ###################################################

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

    # Saving final rectified crop
    
    rectified_crop = cv2.rotate(polar_image[int(min(warped_corners[1])):int(max(warped_corners[1])),int(min(warped_corners[0])):int(max(warped_corners[0]))], cv2.ROTATE_90_COUNTERCLOCKWISE)
    name = "result_" + filename
    
    end_time = timeit.default_timer()
    print(f"Elapsed time: {end_time - start_time}")
    #cv2.imwrite(name, rectified_crop)
    
except Exception as e:
    print(f"An error occurred:\n\t{e}")
    sys.exit(2)