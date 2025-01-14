import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

import atomai as aoi

from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.spatial.distance import cdist


def load_images_from_folder(folder_path):
    images = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") ]
    
    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(folder_path, filename)
        img = np.loadtxt(img_path)
        normalizedData = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        normalizedData = cv.resize(normalizedData, (512,512), interpolation=cv.INTER_LINEAR)
        images.append(normalizedData)
    return images

def get_training_data(image, window_size):

    training_data = []
    training_coordinates = []
    
    img_coords = get_coordinates(image)
    img_coords[:,[1,0]] = img_coords[:,[0,1]]
    crops = aoi.utils.get_imgstack(image, img_coords, window_size)
    for i in range(len(crops[0])):
        clipped_image = intensity_clipping(crops[0][i], 0.999, window_size)
        cropped_image = crop_image(clipped_image)
        training_data.append(cropped_image)
        training_coordinates.append(crops[1][i])
        
    training_data = np.array(training_data)
    training_coordinates = np.array(training_coordinates)

    return(training_data, training_coordinates)

def get_coordinates(image):
    threshData = cv.adaptiveThreshold(image, 110, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY,25,-29)
    contours, hierarchy = cv.findContours(threshData, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    coordinates = []
    match_shapes = []
    circle_mask = np.zeros(image.shape)
    circle = cv.circle(circle_mask,(150,150),6,255,2)
    
    for i, count in enumerate(contours):
    
        area = cv.contourArea(count)
        x,y,w,h = cv.boundingRect(count)
        rect_area = w*h
        extent = float(area)/rect_area
        ret = cv.matchShapes(circle,count,1,0.0)
        match_shapes.append(ret)
    
        if extent<0.3:
            match_shapes[i] = 0
        
        if extent > 0.3:  #Filters out edges
            mask = np.zeros(image.shape)
            cv.drawContours(mask, [count], -1, (0, 255, 0), 1)
            kpCnt = len(count)
            x = 0
            y = 0
        
            for kp in count:
                x = x+kp[0][0]
                y = y+kp[0][1]
    
            coordinates.append(np.array([i,x/kpCnt, y/kpCnt]))
    
    coordinates = np.stack(coordinates)
    match_shapes = np.asarray(match_shapes)
    
    for i, shape in enumerate(match_shapes):
        if shape > 1.1:
            coordinates = coordinates[coordinates[:,0] != i]
    
    coordinates = coordinates[:,1:]
    
    for i, count in enumerate(contours):
        if match_shapes[i] > 1.1:
            dimer_mask = np.zeros(image.shape)
            cv.drawContours(dimer_mask, [contours[i]], -1, 1, -1)
            region_of_interest = image * dimer_mask
            neighborhood = generate_binary_structure(2,2)
            local_max = maximum_filter(region_of_interest, footprint=neighborhood)==region_of_interest
            background = (region_of_interest==0)
            eroded_background = binary_erosion(background, structure = neighborhood, border_value =1)
            detected_peaks = local_max ^ eroded_background
            if np.where(detected_peaks == True)[0].shape[0] > 1:
                peak_coords_list = []
                for j in range(np.where(detected_peaks == True)[0].shape[0]):
                    peak_coords = np.array([[np.where(detected_peaks == True)[1][j],
                                             np.where(detected_peaks == True)[0][j]]])
                    peak_coords_list.append(peak_coords)
    
                peak_coords_list = np.asarray(peak_coords_list).reshape(np.where(detected_peaks == True)[0].shape[0],2)
    
                distances = cdist(peak_coords_list, peak_coords_list)
                close_peaks = np.where((distances <= 4) & (distances != 0))
                faulty_peaks = np.where((distances <= 1) & (distances != 0))
                
                peaks_to_remove = set()
    
                if peak_coords_list.shape[0] > 2:
                    for i, j in zip(close_peaks[0], close_peaks[1]):
                        if i < j:  
                            peaks_to_remove.add(j)
                
                else:
                    for i,j in zip(faulty_peaks[0], faulty_peaks[1]):
                        if i < j:
                            peaks_to_remove.add(j)
    
                peak_coords_list = np.delete(peak_coords_list, list(peaks_to_remove), axis=0)
    
                coordinates = np.concatenate((coordinates, peak_coords_list), axis=0)
            else:
                single_peak_coords = np.array([np.where(detected_peaks == True)[1],
                                             np.where(detected_peaks == True)[0]]).reshape(1,2)
                coordinates = np.concatenate((coordinates, single_peak_coords), axis=0)
    
    return(coordinates)

def crop_image(image):
    center = np.array([image.shape[0]//2, image.shape[1]//2])
    mask = np.zeros((32, 32))
    y, x = np.ogrid[:32, :32]
    distance_from_point = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    circle_mask = distance_from_point <= 5
    mask[circle_mask] = 1
    cropped_array = np.zeros((32, 32))
    cropped_array[circle_mask] = mask[circle_mask] 
    cropped_image = cropped_array*image
    return cropped_image

def intensity_clipping(image, clip, window_size):
    max_allowed = np.quantile(image, q=clip)
    image = np.clip(image, a_min=0, a_max=max_allowed)
    background = median_filter(image, window_size)
    clean_image = image - background
    clean_image[clean_image < 0] = 0
    normalized_image = (clean_image - clean_image.min()) / (clean_image.max() - clean_image.min())
    
    return clean_image
