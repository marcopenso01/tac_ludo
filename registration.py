"""
Created on Sat Feb 18 16:07:26 2023

@author: Marco Penso
"""

import scipy
import scipy.io
import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import math
import random
import pydicom
from scipy import ndimage
from skimage import transform
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects

X = []
Y = []

drawing=False # true if mouse is pressed
mode=True

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
  
def imfill(img, dim=None):
    if len(img.shape)==3:
        img = img[:,:,0]
    if dim: 
        img = cv2.resize(img, (dim, dim))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def setDicomWinWidthWinCenter(vol_data, winwidth, wincenter):
    vol_temp = np.copy(vol_data)
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    
    vol_temp = ((vol_temp[:]-min)*dFactor).astype('int16')

    min_index = vol_temp < 0
    vol_temp[min_index] = 0
    max_index = vol_temp > 255
    vol_temp[max_index] = 255

    return vol_temp

def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny), dtype=img.dtype)
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped

def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1)), dtype=img.dtype), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y), dtype=img.dtype), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512), dtype=img.dtype), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y), dtype=img.dtype), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
def rotate_image(slice, angle, interp=cv2.INTER_LINEAR):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):      
        img = stack[i]
        rows, cols = img.shape[:2]
        
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)
        
        if len(stack)>1:
            RGB.append(img_rot)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return img_rot
    
def measure_angle(pointA, pointB, pointC, pointD):
    # point is a tuple of shape (x,y)
    # return angle in degree
    
    # Calculate vector by initial and terminal points:
    AB = (pointB[0]-pointA[0],pointB[1]-pointA[1])
    CD = (pointD[0]-pointC[0],pointD[1]-pointC[1])
    # Calculate dot product:
    dot = (AB[0]*CD[0]) + (AB[1]*CD[1])
    # Calculate magnitude of a vector:
    magAB = math.sqrt((AB[0]**2) + (AB[1]**2))
    magCD = math.sqrt((CD[0]**2) + (CD[1]**2))
    # Calculate angle between vectors:
    cos_alpha = dot / (magAB*magCD)
    return math.degrees(math.acos(cos_alpha))%360

def measure_dist(pointA, pointB):
    # point is a tuple of shape (x,y)
    # return distance between pointA and pointB
    return math.sqrt(((pointA[0]-pointB[0])**2) + ((pointA[1]-pointB[1])**2))
    
def rot_scale_img(slice, angle=0, scale_vector=None, size=None, anti_aliasing=None, crop=True):
    if angle != 0:
        img = rotate_image(slice, angle)
    dim = slice.shape[0]
    if scale_vector:
        if anti_aliasing==None:
            if scale_vector <1: 
                anti_aliasing=True
            else:
                anti_aliasing=False
        img_scaled = transform.rescale(img,
                                       scale_vector,
                                       order=1,
                                       preserve_range=True,
                                       multichannel=False,
                                       anti_aliasing=anti_aliasing,
                                       mode='constant')
    if size:
        if anti_aliasing==None:
            if size < dim:
                anti_aliasing=True
            else:  
                anti_alisting=False
        img_scaled = transform.resize(img,
                                      (size, size),
                                      order=1,
                                      preserve_range=True,
                                      anti_aliasing=anti_aliasing)
    if crop:
        return crop_or_pad_slice_to_size(img_scaled, slice.shape[0], slice.shape[1]).astype(slice.dtype)
    else:
        return img_scaled.astype(slice.dtype)
    
def translate_img(slice, dx, dy):
    dx=int(dx)
    dy=int(dy)
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
    
        if dx < 0:
            img = np.append(img[abs(dx):,:], np.zeros((abs(dx), y), dtype=img.dtype), axis=0)
        if dx > 0:
            img = np.append(np.zeros((dx, y), dtype=img.dtype), img[:-dx:,:], axis=0)
        if dy < 0:
            img = np.append(img[:, abs(dy)::], np.zeros((x, abs(dy)), dtype=img.dtype), axis=1)
        if dy > 0:
            img = np.append(np.zeros((x,dy), dtype=img.dtype), img[:,:-dy], axis=1)
    
        if len(stack)>1:
            RGB.append(img)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return img
    
def dc(result, reference):
    """
    Dice coefficient
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
        
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

def roi_(img):    
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_left_x = 1000
    top_left_y = 1000
    bottom_right_x = 0
    bottom_right_y = 0
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if x < top_left_x:
            top_left_x = x
        if y < top_left_y:
            top_left_y= y
        if x+w-1 > bottom_right_x:
            bottom_right_x = x+w-1
        if y+h-1 > bottom_right_y:
            bottom_right_y = y+h-1
    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)
    #print('top left=',top_left)
    #print('bottom right=',bottom_right)
    cx = int((top_left[1]+bottom_right[1])/2)   #row
    cy = int((top_left[0]+bottom_right[0])/2)   #column
    nx = int(bottom_right[1]-top_left[1])
    ny = int(bottom_right[0]-top_left[0])
    return cx, cy, nx, ny
    
    
input_folder = 'F:/CT-tesi/RM'
patient = 18

input_folder = os.path.join(input_folder, 'paz' + str(patient))
            
path_art = os.path.join(input_folder, 'ct', 'ART')

vol_art = []
for i in range(len(os.listdir(path_art))):
    dcmPath = os.path.join(path_art, os.listdir(path_art)[i])
    data_row_img = pydicom.dcmread(dcmPath)
    vol_art.append(data_row_img.pixel_array)

vol_art = np.asarray(vol_art)

#set short-axis
vol_art = vol_art.transpose([2,1,0])
'''
# select visually the first and last slice
for i in range(len(vol_art)):
    plt.figure()
    plt.imshow(vol_art[i])
    plt.title(i)
'''
first_el = 270
last_el= 442
vol_art = list(vol_art)
del vol_art[0:first_el]
del vol_art[last_el-first_el::]

# select the basal and apical slice in RM
basal_slice = 3
apical_slice = 12

path_rm = r'F:/CT-tesi/RM/paz18/rm'
path_rm_giu = os.path.join(path_rm, 'seg_giu')
path_rm_giu = os.path.join(path_rm_giu, os.listdir(path_rm_giu)[0])
path_rm_myo = os.path.join(path_rm, 'seg_myo')
path_rm_myo = os.path.join(path_rm_myo, os.listdir(path_rm_myo)[0])
path_rm_scar = os.path.join(path_rm, 'seg_scar')
path_rm_scar = os.path.join(path_rm_scar, os.listdir(path_rm_scar)[0])
path_rm_raw = os.path.join(path_rm, 'raw')
path_rm_raw = os.path.join(path_rm_raw, os.listdir(path_rm_raw)[0])

ct_raw = []
rm_giu = []
rm_myo = []
rm_scar = []
rm_raw = []

if (len(os.listdir(path_rm_myo)) != len(os.listdir(path_rm_giu))) or (len(os.listdir(path_rm_myo)) != len(os.listdir(path_rm_scar))) or (len(os.listdir(path_rm_myo)) != len(os.listdir(path_rm_raw))):
        raise Exception('error number of file: raw %s, myo %s, giu %s, scar %s' % (len(os.listdir(path_rm_raw)), len(os.listdir(path_rm_myo)), len(os.listdir(path_rm_giu)), len(os.listdir(path_rm_scar))))

step = int(len(vol_art) / (apical_slice-basal_slice))
dim = pydicom.dcmread(os.path.join(path_rm_myo, os.listdir(path_rm_myo)[0])).pixel_array.shape[0]
print('dim:', dim)

for i in range(basal_slice-1,apical_slice):
    flag=0
    dcmPath = os.path.join(path_rm_myo, os.listdir(path_rm_myo)[i])
    data_row_img = pydicom.dcmread(dcmPath)
    img = data_row_img.pixel_array
    img = crop_or_pad_slice_to_size(img, dim-70, dim-70)
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            if img[r,c,0] != img[r,c,1] or img[r,c,0] != img[r,c,2]:
                flag=1
                # save mask myo
                temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                for rr in range(0, img.shape[0]):
                    for cc in range(0, img.shape[1]):
                        if img[rr,cc,0] != img[rr,cc,1] or img[rr,cc,0] != img[rr,cc,2]:
                            temp_img[rr,cc]=255
                mask = imfill(temp_img)
                mask[mask > 0] = 1
                mask = crop_or_pad_slice_to_size(mask, dim, dim)
                rm_myo.append(mask)
                # save giu
                img = pydicom.dcmread(os.path.join(path_rm_giu, os.listdir(path_rm_giu)[i])).pixel_array
                img = crop_or_pad_slice_to_size(img, dim-70, dim-70)
                img = crop_or_pad_slice_to_size(img, dim, dim)
                rm_giu.append(img)
                # save mask scar
                img = pydicom.dcmread(os.path.join(path_rm_scar, os.listdir(path_rm_scar)[i])).pixel_array
                img = crop_or_pad_slice_to_size(img, dim-70, dim-70)
                temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                for rr in range(0, img.shape[0]):
                    for cc in range(0, img.shape[1]):
                        if img[rr,cc,0] != img[rr,cc,1] or img[rr,cc,0] != img[rr,cc,2]:
                            temp_img[rr,cc]=255
                mask = imfill(temp_img)
                mask[mask > 0] = 1
                mask = crop_or_pad_slice_to_size(mask, dim, dim)
                rm_scar.append(mask)
                # save raw
                img = pydicom.dcmread(os.path.join(path_rm_raw, os.listdir(path_rm_raw)[i])).pixel_array
                img = img.astype(np.uint16)
                if img.shape[0] != dim:
                    img = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_CUBIC)
                img = crop_or_pad_slice_to_size(img, dim-70, dim-70)
                img = crop_or_pad_slice_to_size(img, dim, dim)
                rm_raw.append(img)
                # save ct
                img = vol_art[(i-(basal_slice-1))*step]
                img = crop_or_pad_slice_to_size(img, max(img.shape), max(img.shape))
                ct_raw.append(img)
            if flag:
                break
        if flag:
            break
    
# first RIGID REGISTRATION
# reference point
pt_rm_giu = {'sup':[], 'inf':[]}
pt_ct_giu = {'sup':[], 'inf':[]}
ct_sup_pt_map = []
ct_myo = []
myo_ = {'rm_c':[], 'rm_l':[], 'ct_c':[], 'ct_l':[]}

for i in range(len(rm_giu)):
    # punto giunzione superiore
    temp_img = np.zeros((dim,dim), dtype=np.uint8)
    for r in range(0, dim):
        for c in range(0, dim):
            if rm_giu[i][r,c,0] == rm_giu[i][r,c,1] and rm_giu[i][r,c,0] != rm_giu[i][r,c,2]:
                temp_img[r,c]=255
    pt_rm_giu['sup'].append((int(round(ndimage.measurements.center_of_mass(temp_img)[0])),int(round(ndimage.measurements.center_of_mass(temp_img)[1]))))
    # punto di giunzione inferiore
    temp_img = np.zeros((dim,dim), dtype=np.uint8)
    for r in range(0, dim):
        for c in range(0, dim):
            if rm_giu[i][r,c,0] == rm_giu[i][r,c,2] and rm_giu[i][r,c,0] != rm_giu[i][r,c,1]:
                temp_img[r,c]=255
    pt_rm_giu['inf'].append((int(round(ndimage.measurements.center_of_mass(temp_img)[0])),int(round(ndimage.measurements.center_of_mass(temp_img)[1]))))

# select superior and inferior point 
tit=['sup point', 'inf point']
for i in range(len(ct_raw)):
    for j in range(2):
        print('---select the %s' % tit[j])
        X = []
        Y = []
        img = cv2.normalize(src=ct_raw[i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("image", img)
        cv2.namedWindow('image')
        cv2.setMouseCallback("image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if j==0:
            pt_ct_giu['sup'].append((X[0],Y[0]))
            img = np.zeros((dim,dim), dtype=np.uint8)
            img[X[0],Y[0]]=255
            ct_sup_pt_map.append(img)
        elif j==1:
            pt_ct_giu['inf'].append((X[0],Y[0]))
            
# segmentation of the ct myo
tit=['epicardium', 'endocardium']
for i in range(len(ct_raw)):
    print("{}/{}".format(i+1, len(ct_raw)))
    print('---Segmenting ct myocardium:')
    for ii in range(2):
        img = ct_raw[i].copy()
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit = 1.5)
        img = clahe.apply(img)
        img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
        image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        cv2.namedWindow(tit[ii])
        cv2.setMouseCallback(tit[ii],paint_draw)
        while(1):
            cv2.imshow(tit[ii],img)
            k=cv2.waitKey(1)& 0xFF
            if k==27: #Escape KEY
                if ii==0:   
                    epi = imfill(image_binary, dim)
                    epi[epi>0]=1                    
                elif ii==1:                                         
                    endo = imfill(image_binary, dim)
                    endo[endo>0]=1
                break
        cv2.destroyAllWindows()  
    mask = epi - endo
    ct_myo.append(mask)
    
'''    
# plot segmentation
for i in range(len(ct_myo)):
    plt.figure()
    plt.imshow(ct_myo[i])
    plt.title(i)
'''

# based on the segmentation, save the center and len_max of myo
for t, n in zip([rm_myo, ct_myo], range(2)):
    for i in range(len(rm_myo)):
        cx, cy, nx, ny = roi_(t[i])
        n_max = max(nx, ny)+50
        if n == 0:
            #myo_['rm_c'].append((cx,cy))
            myo_['rm_l'].append(n_max)
        if n == 1: 
            #myo_['ct_c'].append((cx,cy))
            myo_['ct_l'].append(n_max)

ct_rigid_trans = []
for i in range(len(ct_raw)):
    angle = round(measure_angle(pt_rm_giu['sup'][i], pt_rm_giu['inf'][i], pt_ct_giu['sup'][i], pt_ct_giu['inf'][i]), 1)
    # scale vector based on point of junction points
    #dist_rm = measure_dist(pt_rm_giu['sup'][i], pt_rm_giu['inf'][i])
    #dist_ct = measure_dist(pt_ct_giu['sup'][i], pt_ct_giu['inf'][i])
    #scale_vector = dist_rm/dist_ct
    # scale vector based on size of myo
    scale_vector = myo_['rm_l'][i]/myo_['ct_l'][i]
    # rotate e rescale the ct img
    #img = rot_scale_img(ct_raw[i], angle, scale_vector, crop=False)
    
    # rotate and resize image based on size of myocardium
    # this is more precise that resize according to juntion points
    img = rot_scale_img(ct_raw[i], angle, scale_vector=scale_vector, crop=True)
    mask = rot_scale_img(ct_myo[i], angle, scale_vector=scale_vector, anti_aliasing=False, crop=True)
    
    '''
    # creo una maschera per il sup point ct, e lo ruoto e rescale
    # calcolo la distanza in x e y tra il super point tac e rm
    sup_point_img = rot_scale_img(ct_sup_pt_map[i], angle, scale_vector=scale_vector, crop=True)
    sup_pt = np.argwhere(sup_point_img==(sup_point_img.max()))
    x = pt_rm_giu['sup'][i][0]-sup_pt[0][0]
    y = pt_rm_giu['sup'][i][1]-sup_pt[0][1]
    '''
    # translate based on the center of myo
    # first define center of myo for rm and ct
    cx, cy, _, _ = roi_(rm_myo[i])
    myo_['rm_c'].append((cx,cy))
    cx, cy, _, _ = roi_(mask)
    myo_['ct_c'].append((cx,cy))
            
    x = myo_['rm_c'][i][0]-myo_['ct_c'][i][0]
    y = myo_['rm_c'][i][1]-myo_['ct_c'][i][1]
    
    img = translate_img(img,x,y)
    mask = translate_img(mask,x,y)

    # save ct after rigid registration
    ct_rigid_trans.append(img.astype(np.int16))
    ct_myo[i]=mask
    
'''
# Show the images by stacking them left-right with hstack 
for i in range(len(ct_rigid_trans)):
    plt.figure()
    plt.imshow(np.hstack((rm_raw[i], ct_rigid_trans[i])))
'''

# crop images
rm_raw_crop=[]
rm_myo_crop=[]
rm_scar_crop=[]
ct_raw_crop=[]
ct_myo_crop=[]
for i in range(len(ct_rigid_trans)):
    rm_raw_crop.append(crop_or_pad_slice_to_size_specific_point(rm_raw[i], myo_['rm_l'][i], myo_['rm_l'][i], myo_['rm_c'][i][0], myo_['rm_c'][i][1]))
    rm_myo_crop.append(crop_or_pad_slice_to_size_specific_point(rm_myo[i], myo_['rm_l'][i], myo_['rm_l'][i], myo_['rm_c'][i][0], myo_['rm_c'][i][1]))
    rm_scar_crop.append(crop_or_pad_slice_to_size_specific_point(rm_scar[i], myo_['rm_l'][i], myo_['rm_l'][i], myo_['rm_c'][i][0], myo_['rm_c'][i][1]))
    ct_raw_crop.append(crop_or_pad_slice_to_size_specific_point(ct_rigid_trans[i], myo_['rm_l'][i], myo_['rm_l'][i], myo_['rm_c'][i][0], myo_['rm_c'][i][1]))
    ct_myo_crop.append(crop_or_pad_slice_to_size_specific_point(ct_myo[i], myo_['rm_l'][i], myo_['rm_l'][i], myo_['rm_c'][i][0], myo_['rm_c'][i][1]))

'''
for i in range(len(ct_raw_crop)):
    plt.figure()
    plt.imshow(np.hstack((rm_raw_crop[i], ct_raw_crop[i])))
'''
i=1
mask_rm = rm_myo_crop[i]
mask_rm[mask_rm>0]=255
mask_ct = imfill(ct_myo_crop[i])
img_pre = ct_raw_crop[i].copy() # Image to be aligned.
img_post = rm_raw_crop[i].copy() # Reference image.
#remove negative values
img_pre = img_pre + abs(img_pre.min())
img_pre[mask_ct==0]=0
img_post[mask_rm==0]=0
# convert to uint8
img_pre_8 = cv2.normalize(src=img_pre, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img_post_8 = cv2.normalize(src=img_post, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
height, width = img_pre.shape
#Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(nfeatures=5000,edgeThreshold=0)
#Find keypoints and descriptors.
#The first arg is the image, second arg is the mask which is not required in this case.
kp1, d1 = orb_detector.detectAndCompute(img_pre_8, None)
kp2, d2 = orb_detector.detectAndCompute(img_post_8, None)
#plot
img1_kp = cv2.drawKeypoints(img_pre_8, kp1, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DEFAULT)
img2_kp = cv2.drawKeypoints(img_post_8, kp2, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DEFAULT)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(img1_kp)
ax2 = fig.add_subplot(122)
ax2.imshow(img2_kp)
plt.show()

#Match features between the two images.
#We create a Brute Force matcher with Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

#Match the two sets of descriptors.
matches = matcher.match(d1, d2)

#Sort matches on the basis of their Hamming distance.
matches.sort(key = lambda x: x.distance)

#Take the top 90 % matches forward.
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)

#Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for j in range(len(matches)):
    p1[j, :] = kp1[matches[j].queryIdx].pt
    p2[j, :] = kp2[matches[j].trainIdx].pt

#Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

#Use this matrix to transform the colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img_pre, homography, (width, height))
    
fig = plt.figure()
ax1 = fig.add_subplot(131)
plt.title('img rm')
ax1.imshow(img_post)
ax2 = fig.add_subplot(132)
plt.title('img ct')
ax2.imshow(img_pre)
ax3 = fig.add_subplot(133)
plt.title('img ct transf')
ax3.imshow(transformed_img)
plt.show()
