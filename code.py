import os
import copy
import h5py
import datetime
import numpy as np
import os, time, cv2
from osgeo import gdal
import scipy.io as sio
from skimage import measure, color
from scipy.io import loadmat
from osgeo import gdal_array
from osgeo import ogr, osr, gdal

tree_cover_file = r"tree_cover.tif"
dem_file = r"dem.tif"
mountain_file = r"mountain_Resample1.tif"

def read_tif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    return dataset, im_height, im_width, im_bands, im_geotrans, im_proj

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

hansen_dataset, hansen_height, hansen_width, hansen_bands, hansen_geotrans, hansen_proj = read_tif(tree_cover_file)
tree_cover_data = hansen_dataset.ReadAsArray()

dem_dataset, dem_height, dem_width, dem_bands, dem_geotrans, dem_proj = read_tif(dem_file)
dem_data = dem_dataset.ReadAsArray()

mountain_dataset, mountain_height, mountain_width, mountain_bands, mountain_geotrans, mountain_proj = read_tif(mountain_file)
mountain_data = mountain_dataset.ReadAsArray()

tree_cover_data[tree_cover_data <= 5] = 1
tree_cover_data[tree_cover_data > 5] = 0
tree_cover_data[tree_cover_data == 1] = 255
tree_cover_data = tree_cover_data.astype(np.uint8)

labels = measure.label(tree_cover_data, connectivity=2)

mountain_top_x = mountain_geotrans[0]
mountain_top_y = mountain_geotrans[3]

x_resolution = mountain_geotrans[1]
y_resolution = mountain_geotrans[5]

m, n = dem_data.shape
index = int(dem_data.argmax())
x = int(index / n)
y = index % n
print(dem_data[x, y])
region_label = labels[x, y]
labels[labels != region_label] = -1
labels[labels == region_label] = 255
labels[labels == -1] = 0
labels = labels.astype(np.uint8)

out = FillHole(labels)
a = cv2.Canny(out, 0, 255)
cv2.imwrite('canny.tif', a)























