# Multiple landmark detection in 3D ultrasound images of fetal head
# Network inference with evaluations
#
# Reference
# Fast Multiple Landmark Localisation Using a Patch-based Iterative Network
# https://arxiv.org/abs/1806.06987
#
# Code EDITED BY: Xingchen Xiao
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.ndimage
import time
import torch
from utils import input_data_torch, shape_model_func, patch, save, visual
from network_torch import VisitNet
np.random.seed(0)

class Config(object):
    """Inference configurations."""
    # File paths
    data_dir = './data_2d/Images'
    label_dir = '../landmark_detection_PIN/data_2d/Landmarks_old'
    # label_dir = '../landmark_detection_PIN/data_2d/landmarks'
    train_list_file = './data_2d/train_260/list_train.txt'
    test_list_file = './data_2d/train_260/list_test.txt'
    # train_list_file = './data_2d/list_train.txt'
    # test_list_file = './data_2d/list_test.txt'
    model_dir = './cnn_model'
    # Shape model parameters
    shape_model_file = './shape_model/shape_model_260_3/ShapeModel.mat'
    shape_model_file = './shape_model/shape_model_260_3/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = 3   # Number of landmarks
    landmark_unwant = []     # list of unwanted landmark indices
    # Testing parameters
    box_size = 121          # patch size (odd number)
    max_test_steps = 5     # Number of inference steps
    num_random_init = 3    # Number of random initialisations used
    predict_mode = 1        # How the new patch position is computed.
                            # 0: Classification and regression. Hard classification
                            # 1: Classification and regression. Soft classification. Multiply classification probabilities with regressed distances
                            # 2: Regression only
                            # 3: Classification only
    # Visualisation parameters
    visual = True           # Whether to save visualisation
    dropout = 0.8

def main():
    config = Config()

    # Load shape model
    shape_model = shape_model_func.load_shape_model(config.shape_model_file, config.eigvec_per)
    num_cnn_output_c = 2 * shape_model['Evectors'].shape[1]
    num_cnn_output_r = shape_model['Evectors'].shape[1]

    # Load images and landmarks
    data = input_data_torch.read_data_sets(config.data_dir,
                                     config.label_dir,
                                     config.train_list_file,
                                     config.test_list_file,
                                     config.landmark_count,
                                     config.landmark_unwant,
                                     shape_model)

    print("Start inference...")
    model = VisitNet(num_cnn_output_c,num_cnn_output_r,config.dropout)

    model.load_state_dict(torch.load('../landmark_detection_PIN/cnn_model/model2_train_260_99999.pt'))
    model.eval()

    with torch.no_grad():
        # # Evaluation on test-set
        predict(data[1], config, shape_model, False,
                model)


        # # # Evaluation on train-set
        # predict(data[0], config, shape_model, True,
        #         model)



def predict(data, config, shape_model, train,
            model):
    """Find the path of the landmark iteratively, and evaluate the results.

    Args:
      data: dataset
      config: testing parameters
      shape_model: structure containing shape model
      train: True: train or False: test dataset
      sess, x, action_ind, yc, yr, keep_prob: tensorflow nodes required for inference

    """

    images = data.images
    landmarks_gt = data.labels
    names = data.names
    pix_dim = data.pix_dim
    num_landmarks = config.landmark_count
    img_count = len(images)
    max_test_steps = config.max_test_steps
    num_examples = config.num_random_init + 1

    landmarks_all_steps = np.zeros((img_count, max_test_steps + 1, num_examples, num_landmarks, 3))
    landmarks_mean = np.zeros((img_count, num_landmarks, 3), dtype=np.float32)
    landmarks_mean_unscale = np.zeros((img_count, num_landmarks, 3), dtype=np.float32)
    landmarks_gt_unscale = np.zeros((img_count, num_landmarks, 3), dtype=np.float32)
    images_unscale = []
    time_elapsed = np.zeros(img_count)

    for i in range(img_count):
        # Predict landmarks iteratively
        start_time = time.time()
        landmarks_all_steps[i], landmarks_mean[i] = predict_landmarks(images[i], config, shape_model,
                      model)
        end_time = time.time()
        time_elapsed[i] = end_time - start_time

        # Convert the scaling back to that of the original image.
        landmarks_mean_unscale[i] = landmarks_mean[i] * pix_dim[i, :]
        landmarks_gt_unscale[i] = landmarks_gt[i] * pix_dim[i, :]
        images_unscale.append(scipy.ndimage.zoom(images[i][:, :, :, 0], pix_dim[i]))

        print("Image {}/{}: {}, time = {:.10f}s".format(i+1, img_count, names[i], time_elapsed[i]))

    # Time
    time_elapsed_mean = np.mean(time_elapsed)
    print("Mean running time = {:.10f}s\n".format(time_elapsed_mean))

    # Evaluate distance error
    err, err_mm = compute_err(landmarks_mean, landmarks_gt, pix_dim)

    # Save distance error to txt file
    save.save_err('./results/dist_err', train, names, err, err_mm)

    # Save predicted landmarks as txt files. Landmarks are in voxel coordinates. Not in CNN coordinates.
    save.save_landmarks('./results/landmarks', train, names, landmarks_mean_unscale, config.landmark_unwant)

    # Visualisation
    # if config.visual:
    #     print("Show visualisation...")
    #     for i in range(img_count):
    #         print("Processing visualisation {}/{}: {}".format(i+1, img_count, names[i]))
    #         # visual.plot_landmarks_2d('./results/landmarks_visual2D', train, names[i], images_unscale[i],
    #         #                          landmarks_mean_unscale[i], landmarks_gt_unscale[i])
    #         visual.plot_landmarks_3d('./results/landmarks_visual3D', train, names[i], landmarks_mean[i],
    #                                  landmarks_gt[i], images[i].shape)
    #         visual.plot_landmarks_path('./results/landmark_path', train, names[i], landmarks_all_steps[i],
    #                                    landmarks_gt[i], images[i].shape)


def predict_landmarks(image, config, shape_model,
                      model):
    """Predict landmarks iteratively.

    Args:
      image: image with dimensions=[width, height, depth, channel].
      config: test parameters
      shape_model: structure containing shape model
      sess, x, action_ind, yc, yr, keep_prob: tensorflow nodes required for inference

    Returns:
      landmarks_all_steps: predicted landmarks. [max_test_steps + 1, num_examples, num_landmarks, 3]
      landmarks_mean: mean predicted landmarks across all num_examples. [num_landmarks, 3]

    """

    num_landmarks = config.landmark_count
    max_test_steps = config.max_test_steps
    box_size = config.box_size
    box_r = int((box_size-1)/2)

    # Initialise shape parameters, b=0 and landmarks, x
    b = shape_model_func.init_shape_params(config.num_random_init, None, config.sd, shape_model)     # b=[num_examples, num_shape_params]
    landmarks = shape_model_func.b2landmarks(b, shape_model)    # landmarks=[num_examples, num_landmarks, 3]
    num_examples = b.shape[0]

    # Extract patches from landmarks
    patches = np.zeros((num_examples, box_size, box_size, 3*num_landmarks))
    for j in range(num_examples):
        patches[j] = patch.extract_patch_all_landmarks(image, landmarks[j], box_r)       # patches=[num_examples, box_size, box_size, 3*num_landmarks]

    landmarks_all_steps = np.zeros((max_test_steps + 1, num_examples, num_landmarks, 3))
    landmarks_all_steps[0] = landmarks

    patches = torch.from_numpy(patches).permute(0, 3, 1, 2).to(torch.float32)

    for j in range(max_test_steps):    # find path of landmark iteratively

        # Predict CNN outputs
        yc_val, yr_val,_ = model(patches)

        yc_val = yc_val.detach().numpy()
        yr_val= yr_val.detach().numpy()

        # Compute classification probabilities
        action_prob = np.exp(yc_val - np.expand_dims(np.amax(yc_val, axis=1), 1))
        action_prob = action_prob / np.expand_dims(np.sum(action_prob, axis=1), 1)      # action_prob=[num_examples, 2*num_shape_params]

        # Update b values by combining classification and regression outputs
        b = update_b(b, action_prob, yr_val, config.predict_mode)

        # Convert b to landmarks
        landmarks = shape_model_func.b2landmarks(b, shape_model)    # landmarks=[num_examples, num_landmarks, 3]
        landmarks_all_steps[j+1] = landmarks

        # patches = patches.permute(0, 1, 2, 3)
        # Extract patches from landmarks
        for k in range(num_examples):
            patches[k] = torch.from_numpy(patch.extract_patch_all_landmarks(image, landmarks[k], box_r)).permute(2,0,1)
             # patches=[num_examples, box_size, box_size, 3*num_landmarks]

    # Compute mean of all initialisations
    landmarks_mean = np.mean(landmarks_all_steps[-1, :, :, :], axis=0)  # landmarks_mean=[num_landmarks, 3]

    return landmarks_all_steps, landmarks_mean


def update_b(b, action_prob, yr_val, predict_mode):
    """Update new shape parameters b using the regression and classification output.

    Args:
      b: current shape parameters values. [num_examples, num_shape_params].
      action_prob: classification output. [num_actions]=[num_examples, 2*num_shape_params]
      yr_val: values of db to regress. yr=b-b_gt. [num_examples, num_shape_params]
      predict_mode: 0: Hard classification. Move regressed distance only in the direction with maximum probability.
                    1: Soft classification. Multiply classification probabilities with regressed distances.
                    2: Regression only.
                    3: Classification only.

    Returns:
      b_new: new b after update. [num_examples, num_shape_params]

    """
    if predict_mode == 0:
        # Hard classification. Move regressed distance only in the direction with maximum probability.
        ind = np.argmax(np.amax(np.reshape(action_prob, (b.shape[0], b.shape[1], 2)), axis=2), axis=1)  # ind = [num_examples]
        row_ind = np.arange(b.shape[0])
        b[row_ind, ind] = b[row_ind, ind] - yr_val[row_ind, ind]
    elif predict_mode == 1:
        # Soft classification. Multiply classification probabilities with regressed distances.
        b = b - yr_val * np.amax(np.reshape(action_prob, (b.shape[0], b.shape[1], 2)), axis=2)
    elif predict_mode == 2:
        # Regression only.
        b = b - yr_val
    elif predict_mode == 3:
        # Classification only
        step = 1
        action_prob_reshape = np.reshape(action_prob, (b.shape[0], b.shape[1], 2))
        ind = np.argmax(np.amax(action_prob_reshape, axis=2), axis=1)   # ind=[num_examples]
        row_ind = np.arange(b.shape[0])
        is_negative = np.argmax(action_prob_reshape[row_ind, ind], axix=1)    # is_negative=[num_examples]
        # Move b in either positive or negative direction
        b[row_ind[is_negative], ind[is_negative]] = b[row_ind[is_negative], ind[is_negative]] + step
        b[row_ind[np.logical_not(is_negative)], ind[np.logical_not(is_negative)]] = b[row_ind[np.logical_not(is_negative)], ind[np.logical_not(is_negative)]] - step

    return b


def compute_err(landmarks, landmarks_gt, pix_dim):
    """Compute the distance error between predicted and GT landmarks.

    Args:
      landmarks: Predicted landmarks [img_count, num_landmarks, 3].
      landmarks_gt: Ground truth landmarks. [img_count, num_landmarks, 3]
      pix_dim: Pixel spacing. [img_count, 3]

    Returns:
      err: distance error in voxel. [img_count, num_landmarks]
      err_mm: distance error in mm. [img_count, num_landmarks]

    """
    num_landmarks = landmarks.shape[1]
    err = np.sqrt(np.sum(np.square(landmarks - landmarks_gt), axis=-1))
    err_mm = np.sqrt(np.sum(np.square((landmarks - landmarks_gt) * pix_dim[:, np.newaxis, :]), axis=-1))
    err_mm_landmark_mean = np.mean(err_mm, axis=0)
    err_mm_landmark_std = np.std(err_mm, axis=0)
    err_mm_mean = np.mean(err_mm)
    err_mm_std = np.std(err_mm)
    str = "Mean distance error (mm): "
    for j in range(num_landmarks):
        str += ("{:.10f} ".format(err_mm_landmark_mean[j]))
    print("{}".format(str))
    str = "Std distance error (mm): "
    for j in range(num_landmarks):
        str += ("{:.10f} ".format(err_mm_landmark_std[j]))
    print("{}".format(str))
    print("Mean distance error (mm) = {:.10f} \nStd distance error (mm) = {:.10f}\n".format(err_mm_mean, err_mm_std))
    return err, err_mm


if __name__ == '__main__':
    main()
