#  Multiple landmark detection in 2D medical cardiac images
#
# Reference
# Fast Multiple Landmark Localisation Using a Patch-based Iterative Network
# https://arxiv.org/abs/1806.06987
#
# Code EDITED BY: Xingchen Xiao
# ==============================================================================


"""Functions for reading input data (image (nifti) and label (txt))."""

import os
import numpy as np
import nibabel as nib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

import shape_model_func

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset
from PIL import Image

from skimage import io
from skimage import transform as sk_transform
from skimage.transform import rescale, resize, downscale_local_mean
import torch

class DataSet(object):
  def __init__(self,
               names,
               images,
               labels,
               shape_params,
               pix_dim):
    assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
    self.num_examples = len(images)
    self.names = names
    self.images = images
    self.labels = labels
    self.shape_params = shape_params
    self.pix_dim = pix_dim


def get_file_list(txt_file):
  """Get a list of filenames.

  Args:
    txt_file: Name of a txt file containing a list of filenames for the images.

  Returns:
    filenames: A list of filenames for the images.

  """
  with open(txt_file) as f:
    filenames = f.read().splitlines()
  return filenames


def extract_image(filename):
  """Extract the image into a 3D numpy array [x, y, z].

  Args:
    filename: Path and name of nifti file.

  Returns:
    data: A 3D numpy array [x, y, z]
    pix_dim: pixel spacings

  """

  img_ = load_image(filename)

  data = image_reshape_to_numpy_v2(img_)
  # data = image_reshape_to_numpy_v2(img_)

  pix_dim =[1,1,1]
  return data, pix_dim


# load image as image object
def load_image(img_path):

    im = io.imread(img_path, as_gray=True)
    
    return im

# load image as image with shape of 101*101, downsized
def load_image_101(img_path):
    
    im = io.imread(img_path, as_gray=True)

    image_resized = resize(im,(101,101), anti_aliasing=True)
    
    return image_resized
    
#load image as image object and using mean-std normalization
def image_reshape_to_numpy(img):
    
    img = np.asarray(img)
    
    img = image_normalized(img)
    
    img_arr = img.reshape(img.shape[0], img.shape[1],1).astype('float32')
    
    return img_arr

# load image as image object and using min-max normalization
def image_reshape_to_numpy_v2(img):
    
    img = np.asarray(img)
    
    img = image_normalized_v2(img)
    
    img_arr = img.reshape(img.shape[0], img.shape[1],1).astype('float32')
    
    return img_arr

#load image as image object and using mean-std normalization
def image_reshape_to_numpy_v3(img):
    
    img = np.asarray(img)
    
    img_arr = img.reshape(img.shape[0], img.shape[1],1).astype('float32')
    
    return img_arr

# mean-std normalization
def image_normalized(ndarray):

    arr_mean = np.mean(ndarray)
    arr_std = np.std(ndarray)
    
    result = (ndarray-arr_mean)/arr_std
    
    return result

#min-max normalization
def image_normalized_v2(ndarray):

    arr_min = np.min(ndarray).astype('float32')
    arr_max = np.max(ndarray).astype('float32')
    
    result = (ndarray.astype('float32')-arr_min)/(arr_max-arr_min)
    
    return result    

  
def extract_label(filename):
  """Extract the labels (landmark coordinates) into a 2D float64 numpy array.

  Args:
    filename: Path and name of txt file containing the landmarks. One row per landmark.

  Returns:
    labels: a 2D float64 numpy array. [landmark_count, 3]
  """
  with open(filename) as f:
    labels = np.empty([0, 3], dtype=np.float64)
    print(labels)
    for line in f:
      a = [float(x) for x in line.split()]
      labels = np.vstack((labels, a))
  return labels


def select_label(labels, landmark_unwant):
  """Unwanted landmarks are removed.
     Remove topHead (landmark index 0).
     Remove left or right ventricle (landmark index (6,7) or (8,9)).
     Remove mid CSP (landmark index 13).
     Remove left and right eyes (landmark index 14 and 15).

  Args:
    labels: a 2D float64 numpy array.
    landmark_unwant: indices of the unwanted landmarks

  Returns:
    labels: a 2D float64 numpy array.
  """
  removed_label_ind = list(landmark_unwant)
  labels = np.delete(labels, removed_label_ind, 0)
  return labels


def extract_all_image_and_label(file_list,
                                data_dir,
                                label_dir,
                                landmark_count,
                                landmark_unwant,
                                shape_model):
  """Load the input images and landmarks and rescale to fixed size.

  Args:
    file_list: txt file containing list of filenames of images
    data_dir: Directory storing images.
    label_dir: Directory storing labels.
    landmark_count: Number of landmarks used (unwanted landmarks removed)
    landmark_unwant: list of unwanted landmark indices
    shape_model: structure containing the shape model

  Returns:
    filenames: list of patient id names
    images: list of img_count 4D numpy arrays with dimensions=[width, height, depth, 1]. Eg. [324, 207, 279, 1]
    labels: landmarks coordinates [img_count, landmark_count, 3]
    shape_params: PCA shape parameters [img_count, shape_param_count]
    pix_dim: mm of each voxel. [img_count, 3]

  """
  filenames = get_file_list(file_list)
  file_count = len(filenames)
  images = []
  labels = np.zeros((file_count, landmark_count, 3), dtype=np.float64)
  pix_dim = np.zeros((file_count, 3))
  for i in range(len(filenames)):
    filename = filenames[i]
    print("Loading image {}/{}: {}".format(i+1, len(filenames), filename))
    # load image
    img, pix_dim[i] = extract_image(os.path.join(data_dir, filename+'.png'))
    # load landmarks and remove unwanted ones. Labels already in voxel coordinate
    label = extract_label(os.path.join(label_dir, filename+'.txt'))
    # label = select_label(label, landmark_unwant)
    # Store extracted data
    images.append(np.expand_dims(img, axis=3))
    labels[i, :, :] = label
  # Compute shape parameters
  shape_params = shape_model_func.landmarks2b(labels, shape_model)
  return filenames, images, labels, shape_params, pix_dim


def read_data_sets(data_dir,
                   label_dir,
                   train_list_file,
                   test_list_file,
                   landmark_count,
                   landmark_unwant,
                   shape_model):
  """Load training and test dataset.

  Args:
    data_dir: Directory storing images.
    label_dir: Directory storing labels.
    train_list_file: txt file containing list of filenames for train images
    test_list_file: txt file containing list of filenames for test images
    landmark_count: Number of landmarks used (unwanted landmarks removed)
    landmark_unwant: list of unwanted landmark indices
    shape_model: structure storing the shape model

  Returns:
    data: A collections.namedtuple containing fields ['train', 'validation', 'test']

  """
  # Load images and landmarks
  print("Loading train images...")
  train_names, train_images, train_labels, train_shape_params, train_pix_dim = extract_all_image_and_label(train_list_file,
                                                                                                           data_dir,
                                                                                                           label_dir,
                                                                                                           landmark_count,
                                                                                                           landmark_unwant,
                                                                                                           shape_model)
  print("Loading test images...")
  test_names, test_images, test_labels, test_shape_params, test_pix_dim = extract_all_image_and_label(test_list_file,
                                                                                                      data_dir,
                                                                                                      label_dir,
                                                                                                      landmark_count,
                                                                                                      landmark_unwant,
                                                                                                      shape_model)
  train = DataSet(train_names, train_images, train_labels, train_shape_params, train_pix_dim)
  test = DataSet(test_names, test_images, test_labels, test_shape_params, test_pix_dim)
  return train, test


