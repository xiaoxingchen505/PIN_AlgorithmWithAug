# Multiple landmark detection in 3D ultrasound images of fetal head
# Network training
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

import os
import sys
import numpy as np
import torch

import torch.nn as nn

from utils import input_data_torch, shape_model_func, network_torch, patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from torch.optim.lr_scheduler import ExponentialLR
from network_torch import VisitNet

class Config(object):
    """Training configurations."""
    # File paths
    data_dir = '../landmark_detection_PIN/data_2d/Images'
    label_dir = '../landmark_detection_PIN/data_2d/landmarks'
    train_list_file = '../landmark_detection_PIN/data_2d/train_280/list_train.txt'
    test_list_file = '../landmark_detection_PIN/data_2d/train_280/list_test.txt'
    log_dir = '../landmark_detection_PIN/logs'
    model_dir = '../landmark_detection_PIN/cnn_model'
    model_file = ''
    # Shape model parameters
    shape_model_file = '../landmark_detection_PIN/shape_model/shape_model_280_4/ShapeModel.mat'
    eigvec_per = 0.995      # Percentage of eigenvectors to keep
    sd = 3.0                # Standard deviation of shape parameters
    landmark_count = 4    # Number of landmarks
    landmark_unwant = []     # list of unwanted landmark indices
    # Training parameters
    resume = False          # Whether to train from scratch or resume previous training
    box_size = 121          # patch size (odd number)
    alpha = 0.5             # Weighting given to the loss (0<=alpha<=1). loss = alpha*loss_c + (1-alpha)*loss_r
    learning_rate = 0.0005
    max_steps = 100000      # Number of steps to train
    save_interval = 25000   # Number of steps in between saving each model
    batch_size = 64       # Training batch size
    dropout = 0.8


def main():
    config = Config()

    # Load shape model
    shape_model = shape_model_func.load_shape_model(config.shape_model_file, config.eigvec_per)
    num_cnn_output_c = 2 * shape_model['Evectors'].shape[1]
    num_cnn_output_r = shape_model['Evectors'].shape[1]

    # Load images and landmarks
    train_data, test_data = input_data_torch.read_data_sets(config.data_dir,
                                        config.label_dir,
                                        config.train_list_file,
                                        config.test_list_file,
                                        config.landmark_count,
                                        config.landmark_unwant,
                                        shape_model)
    
    #intialize weights and bias
    # w = network_torch.network_weights(shape)
    # b = network_torch.bias_variable(shape)
    # Define CNN model
    # net = VisitNet(num_cnn_output_c,num_cnn_output_r,config.dropout)

    net = VisitNet(num_cnn_output_c,num_cnn_output_r,Config.dropout).cuda()

    # loss_c = nn.CrossEntropyLoss()
    # loss_r = nn.MSELoss()

    loss_c = nn.CrossEntropyLoss().cuda()
    loss_r = nn.MSELoss().cuda()


    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    # scheduler = ExponentialLR(optimizer, gamma=0.999)
    
    if config.resume:
        net.load_state_dict(torch.load(config.model_dir+"/"+config.model_file))
        ite_start = config.start_iter
        ite_end = ite_start + config.max_steps

    else:
        ite_start = 0
        ite_end = config.max_steps
    print('Currently Using Cuda mode:',torch.cuda.get_device_name(0))
    for i in range(ite_start, ite_end):
        patches_train, actions_train, dbs_train, _ = get_train_pairs(config.batch_size,
                                                                     train_data.images,
                                                                     train_data.shape_params,
                                                                     config.box_size,
                                                                     num_cnn_output_c,
                                                                     num_cnn_output_r,
                                                                     shape_model,
                                                                     config.sd)
        optimizer.zero_grad()
        
        net.train()

        patches_train_torch = torch.from_numpy(patches_train).permute(0, 3, 1, 2)
       
        actions_train_torch = torch.argmax(torch.from_numpy(actions_train).to(torch.long), dim=1)


        dbs_train_torch = torch.from_numpy(dbs_train).to(torch.float32)

        patches_train_torch = patches_train_torch.cuda()
        actions_train_torch = actions_train_torch.cuda()
        dbs_train_torch = dbs_train_torch.cuda()

        y_c, y_r, _ = net(patches_train_torch)
        loss_c_val = torch.mean(config.alpha * loss_c(y_c, actions_train_torch))
        loss_r_val = torch.mean((1 - config.alpha) * loss_r(y_r, dbs_train_torch))
        # loss_r_val = torch.mean((1 - config.alpha) * torch.pow(dbs_train_torch-y_r,2))

        loss = loss_c_val + loss_r_val
        # print(loss_r_val)
        # loss = config.alpha * loss_c_val + (1 - config.alpha) * loss_r_val

        loss.backward()
        optimizer.step()
        # scheduler.step()
        pred_idx = torch.argmax(y_c, dim=1)
        # print(pred_idx)
        # print()
        # print(actions_train_torch)
        # print()
        # print(pred_idx.eq(actions_train_torch).sum())
        # print()
        # print(pred_idx.eq(actions_train_torch).sum().item())
        # print()

        # print((pred_idx.size()[0]))
        # print()
        # print((pred_idx.size()[0]))

        # if i == 1:
        #    break
        accuracy = pred_idx.eq(actions_train_torch).sum().item() / (pred_idx.size()[0])
        print("[%d/%d] class loss: %f reg loss: %f train loss: %f accuracy: %f" % (i, ite_end, loss_c_val.item(), loss_r_val.item(), loss.item(), accuracy))

        if ((i+1) % config.save_interval) == 0:
            torch.save(net.state_dict(), config.model_dir+"/model2_meanstd_norm_step"+str(i)+".pt")

        if (i+1) % 100 == 0:
            patches_test, actions_test, dbs_test, _ = get_train_pairs(config.batch_size,
                                                                         test_data.images,
                                                                         test_data.shape_params,
                                                                         config.box_size,
                                                                         num_cnn_output_c,
                                                                         num_cnn_output_r,
                                                                         shape_model,
                                                                         config.sd)

            net.eval()
            patches_test_torch = torch.from_numpy(patches_test).permute(0, 3, 1, 2).cuda()
            actions_test_torch = torch.argmax(torch.from_numpy(actions_test).to(torch.long), dim=1).cuda()
            # dbs_test_torch = torch.from_numpy(dbs_test).to(torch.float32)
            y_c_test, _, _ = net(patches_test_torch)
            pred_idx_test = torch.argmax(y_c_test, dim=1)
            # print(pred_idx)
            # print(actions_train_torch)
            # print(pred_idx.eq(actions_train_torch).sum())
            accuracy_test = pred_idx_test.eq(actions_test_torch).sum().item() / (pred_idx_test.size()[0])
            print("[%d/%d] test accuracy: %f" % (i, ite_end, accuracy_test))        



def get_train_pairs(batch_size, images, bs_gt, box_size, num_actions, num_regression_output, shape_model, sd):
    """Randomly sample image patches and corresponding ground truth classification and regression outputs.

    Args:
      batch_size: mini batch size
      images: list of img_count images. Each image is [width, height, depth, channel], [x,y,z,channel]
      bs_gt: Ground truth shape parameters. [img_count, num_shape_params]
      box_size: size of image patch. Scalar.
      num_actions: number of classification outputs
      num_regression_output: number of regression outputs
      shape_model: structure containing shape models
      sd: standard deviation of shape model. Bounds from which to sample bs.

    Returns:
      patches: 2D image patches, [batch_size, box_size, box_size, 3*num_landmarks]
      actions: Ground truth classification output. [batch_size, num_actions], each row is a one hot vector [positive or negative for each shape parameter]
      dbs: Ground truth regression output. [batch_size, num_regression_output]. dbs = bs - bs_gt.
      bs: sampled shape parameters [batch_size, num_regression_output]

    """
    img_count = len(images)
    num_landmarks = 4
    box_r = int((box_size - 1) / 2)
    patches = np.zeros((batch_size, box_size, box_size, int(3*num_landmarks)), np.float32)
    actions_ind = np.zeros(batch_size, dtype=np.uint16)
    actions = np.zeros((batch_size, num_actions), np.float32)

    # get image indices randomly for a mini-batch
    ind = np.random.randint(img_count, size=batch_size)

    # Randomly sample shape parameters, bs
    bounds = sd * np.sqrt(shape_model['Evalues'])
    bs = np.random.rand(batch_size, num_regression_output) * 2 * bounds - bounds

    # Convert shape parameters to landmark
    landmarks = shape_model_func.b2landmarks(bs, shape_model)

    # Extract image patch
    for i in range(batch_size):
        image = images[ind[i]]
        patches[i] = patch.extract_patch_all_landmarks(image, landmarks[i], box_r)

    # Regression values (distances between predicted and GT shape parameters)
    dbs = bs - bs_gt[ind]

    # Extract classification labels as a one-hot vector
    max_db_ind = np.argmax(np.abs(dbs), axis=1)     # [batch_size]
    max_db = dbs[np.arange(dbs.shape[0]), max_db_ind]   # [batch_size]
    is_positive = (max_db > 0)
    actions_ind[is_positive] = max_db_ind[is_positive] * 2
    actions_ind[np.logical_not(is_positive)] = max_db_ind[np.logical_not(is_positive)] * 2 + 1
    actions[np.arange(batch_size), actions_ind] = 1

    return patches, actions, dbs, bs


if __name__ == '__main__':
    main()
