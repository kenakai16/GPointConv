import argparse
import os
import sys
import numpy as np 
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from model.group_pointconv import GroupPointConvDensityClsSsg as GroupPointConvDensityClsSsg
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import math
from random import randint
import random


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('GPointConv')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--gpu', type=str, default='cpu', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--model_name', default='gpointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device =  args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./roration_eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints_rotation/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = 'C:\\Users\\nghiahoang\\Desktop\\OneDrive_1_2-11-2021\\dataset\\modelnet40_normal_resampled'


    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    logger.info("The number of test data is: %d", len(TEST_DATASET))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    num_class = 40
    classifier = GroupPointConvDensityClsSsg(num_classes=num_class).to(device)

    if args.checkpoint is not None:
        print('Load CheckPoint from {}'.format(args.checkpoint))
        logger.info('Load CheckPoint')
        # Load
        checkpoint = torch.load(args.checkpoint, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')
    accuracy = test_original(classifier, testDataLoader)

    logger.info('Total Accuracy: %f'%accuracy)
    logger.info('End of evaluation...')


#--------------------------------------------------------#
def test_original(model, loader):
    device = torch.device('cpu')
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def test_rotation_group(model, loader, split_group = None ,name=None):
    device = torch.device('cpu')

    G24 = torch.from_numpy(np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],

        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [1, 0, 0], [0, -1, 0]],

        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],

        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],

        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

        [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    ])).float()

    if split_group != None:
        r_group = G24[split_group]

    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points[:,:,:3] = torch.matmul(points[:,:,:3], r_group) #rotate-sample
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    print("test_rotation_group:", accuracy)
    return accuracy

def test_random_angel(model, loader, coordinates = "Rx" , phase ="custom"):
    device = torch.device('cpu')
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]

        random.seed(j)

        if phase == "custom":
            r = random.Random(j)
            alpha = r.choice([randint(0, 30),randint(60, 120),randint(150, 180)])
            rotation_angle = alpha*np.pi / 180.

        elif phase == "random":
            alpha = randint(0, 180)
            rotation_angle = alpha*np.pi / 180.

        points[:,:,:3] = rotate_point_cloud_by_angle(points[:,:,:3], coordinates, rotation_angle) #rotate-sample

        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    print("random angel acc:", accuracy)
    return accuracy

def rotate_point_cloud_by_angle(batch_data, coordinates = "Rx" , rotation_angle=np.pi/2):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """

    batch_data = batch_data.cpu().detach().numpy()
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)


        Rx = np.array([[1, 0, 0],
                           [0, cosval, -sinval],
                           [0, sinval, cosval]])

        Ry = np.array([[cosval, 0, sinval],
                           [0, 1, 0],
                           [-sinval, 0, cosval]])

        Rz = np.array([[cosval, -sinval, 0],
                           [sinval, cosval, 0],
                           [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))

        if coordinates=="Rx":
            rotated_matrix = Rx
        elif coordinates=="Ry":
            rotated_matrix = Ry
        elif coordinates=="Rz":
            rotated_matrix = Rz
        else:
            rotated_matrix = R

        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotated_matrix)
    rotated_data= torch.from_numpy(rotated_data)
    return rotated_data

if __name__ == '__main__':
    args = parse_args()
    main(args)
