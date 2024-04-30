import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def compute_LRA(xyz, weighting=False, nsample = 64):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

    eigen_values, vec = M.symeig(eigenvectors=True)

    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    centroid = torch.mean(xyz, dim=1, keepdim=True) #[B, 1, C]
    dist = torch.sum((xyz - centroid) ** 2, -1)
    farthest = torch.max(dist, -1)[1]

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

class ScanObjectNNCls(data.Dataset):

    def __init__(
            self, transforms=None, train=True, self_supervision=False, split='main'
    ):
        super().__init__()

        self.transforms = transforms

        self.self_supervision = self_supervision

        self.train = train

        split = 'main_split' if split=='main' else f'split_{split}'
        root = 'dataset/ScanObjectNN/h5_files/main_split/'
        if self.self_supervision:
            print('self supervision')
            h5 = h5py.File(root + 'training_objectdataset_augmentedrot_scale75.h5', 'r')
            points_train = np.array(h5['data']).astype(np.float32)
            h5.close()
            self.points = points_train
            self.labels = None
        elif train:
            h5 = h5py.File(root + 'training_objectdataset_augmentedrot_scale75.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            h5 = h5py.File(root + 'test_objectdataset_augmentedrot_scale75.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()

        self.class_num = self.labels.max() + 1

#         self.points_ = torch.tensor(self.points).cuda()
#         fps_idx = farthest_point_sample(self.points_[:,:,:3], 1024)
#         self.points_ = index_points(self.points_, fps_idx.long())
#         self.points = self.points_.cpu()
#         del self.points_
#         del fps_idx
#         torch.cuda.empty_cache()

#         norm = torch.zeros_like(self.points)
#         for i in range(self.points.shape[0]):
#             norm_i = compute_LRA(self.points[i,:,:].unsqueeze(0), True, nsample = 32)
#             norm[i,:,:] = norm_i

#         self.points = torch.cat([self.points, norm], dim=-1)

        print('Successfully load ScanObjectNN with', len(self.labels), 'instances')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs][:1024].copy()
        current_points[:, :3] = pc_normalize(current_points[:, :3])
        label = self.labels[idx]
        lra = compute_LRA(torch.from_numpy(current_points[:, :3]).float().unsqueeze(0), True, nsample = 32)

        data = Data(pos=torch.from_numpy(current_points[:, :3]).float(), y=torch.tensor(int(label)).long(),
                    norm=torch.from_numpy(current_points[:, :3]).float(), LRA=lra.squeeze(0).float())
        
        if self.transforms is not None:
            data = self.transforms(data)

        # if self.self_supervision:
        #     return current_points
        # else:
        # label = self.labels[idx]
        return data

        ###############################################################
        # pc = self.data[index][:self.npoints].numpy()
        # # pc = self.data[index].numpy()
        # cls = np.asarray(self.label[index])
        #
        # pc[:, 0:3] = pc_normalize(pc[:, 0:3])
        #
        # points = self._augment_data(pc)  # only shuffle by default

        # print(points)

        # print((self.cache[index] - points).mean())

        # data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long(),
        #             norm=torch.from_numpy(points[:, 3:]).float())
        # data.idx = index
        #
        # if self.transform is not None:
        #     data = self.transform(data)

        # return data

    def __len__(self):
        return self.points.shape[0]

