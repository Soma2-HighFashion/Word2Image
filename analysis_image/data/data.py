try:
    import cPickle as pickle
except:
    import pickle
import os
from PIL import Image
import numpy as np

GEOMETRY = (128, 42)
PATCH_GEOMETRY = (42, 42)

def img2numpy_arr(img_path):
    return np.array(Image.open(img_path))

def generate_patches(ndarr, image_count, patch_count):
	print ndarr.shape

	y,x = image_count
	img_data = np.empty((patch_count*y*x, PATCH_GEOMETRY[0], PATCH_GEOMETRY[1], 3), 
						dtype="float32")
	print y, x, img_data.shape

	patch_index = 0
	for i in range(y):
		crop_y_indices = (i*GEOMETRY[0], (i+1)*GEOMETRY[0])	
		for j in range(x):
			crop_x_indices = (j*GEOMETRY[1], (j+1)*GEOMETRY[1])
			img = ndarr[crop_y_indices[0]:crop_y_indices[1], 
						crop_x_indices[0]:crop_x_indices[1],
						:]

			print crop_y_indices, crop_x_indices

			step = (GEOMETRY[0] - PATCH_GEOMETRY[0]) / (patch_count-1)
			
			patch_image = np.array(
						[img[i*step:i*step+PATCH_GEOMETRY[0], :, : ] for i in range(patch_count)]
						)
			img_data[patch_index*patch_count:(patch_index+1)*patch_count, :, :, :] = patch_image
			patch_index += 1
	img_data /= 255
	return img_data
	
def save2p(data, fname):
    try:
        with open(fname, "wb") as fh:
            pickle.dump(data, fh)
    except IOError as ioerr:
        print ("File Error: %s" % str(ioerr))
    except pickle.PickleError as pklerr:
        print ("Pickle Error: %s" % str(pklerr))

def load(fname):
    savedItems = []
    try:
        with open(fname, "rb") as fh:
            savedItems = pickle.load(fh)
    except IOError as ioerr:
        save2p(savedItems, fname)
    except pickle.PickleError as pklerr:
        print ("Pickle Error: %s" % str(pklerr))
    finally:
        return savedItems
