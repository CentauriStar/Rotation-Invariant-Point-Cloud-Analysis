import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_batch_svd import svd

PI = torch.from_numpy(np.array(np.pi))

def normalize_angle(alpha):
    return alpha / PI

def feat_select(feat, ind):
    assert feat.dim()==3 and ind.dim()==1
    B, C, N = feat.size()
    BNK = ind.size(0)
    K = int(BNK/(B*N))
    base = torch.arange(B, device=feat.device).view(B, 1, 1).repeat(1, 1, N*K) *N

    return torch.gather(feat, 2, (ind.view(B, 1, N*K) - base).repeat(1, C, 1)).view(B, C, N, K)

def knn(x, k, remove_self_loop=True):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    if remove_self_loop:
        idx = pairwise_distance.topk(k=k + 1, dim=-1)[1]  # (batch_size, num_points, k)
        return idx[:, :, 1:]
    else:
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

def get_graph_feature(x, feat=None, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k).cuda()  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    if feat is not None:
        x = feat

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def get_angle(v1, v2, axis=None):
    if axis is None:
        return torch.atan2(
            torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
    else:
        cosine = (v1 * v2).sum(dim=1)
        cross_axis = torch.cross(v1, v2, dim=1)
        sign = torch.ones_like(cosine)
        sign[(cross_axis * axis).sum(dim=1) < 0.] = -1.
        return torch.atan2(
            cross_axis.norm(p=2, dim=1) * sign, cosine)


def point_pair_features(pos_i, pos_j, norm_i, norm_j):
    pseudo = pos_j - pos_i
    return torch.stack([
        pseudo.norm(p=2, dim=1),
        get_angle(norm_i, pseudo),
        get_angle(norm_j, pseudo),
        get_angle(norm_i, norm_j, axis=pseudo)
    ], dim=1)


def get_local_frame_angles(pos_i, pos_j, norm_i, k, axis=None):
    pseudo = pos_j - pos_i
    inplane_direction = torch.cross(norm_i, pseudo) / pseudo.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    '''
    use the projection of center point as the second axis
    '''
    start = torch.cross(norm_i, axis) / axis.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    cross = torch.cross(start, inplane_direction)
    mask = (torch.einsum('md,md->m', cross, norm_i) > 0)
    sins = cross.norm(dim=-1)
    sins[mask] = sins[mask] * (-1)
    coss = torch.einsum('md,md->m', start.squeeze(), inplane_direction)  # m:num_points  k:neigh  d:dim

    angles = torch.atan2(sins, coss)

    return angles.view(-1, 1)


class PaRIConv(nn.Module):
    def __init__(self, in_dim, out_dim, feat_dim=8, k=20):
        super(PaRIConv, self).__init__()
        self.k = k

        self.basis_matrix = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.dynamic_kernel = nn.Sequential(nn.Conv2d(feat_dim, in_dim//2, kernel_size=1), 
                                            nn.BatchNorm2d(in_dim//2),
                                            nn.ReLU(),
                                            nn.Conv2d(in_dim//2, in_dim, kernel_size=1))
        self.act = nn.Sequential(nn.BatchNorm2d(in_dim), nn.ReLU())

        self.edge_conv = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_dim),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, APPF, edge_index, bs):
        _, C = APPF.size()
        APPF = APPF.view(bs, -1, self.k, C).permute(0, 3, 1, 2).contiguous() # [32, 8, 1024, 20]
        APPF = self.dynamic_kernel(APPF) # [32, 32, 1024, 20] in_dim=64,64,128,256
        
        row, col = edge_index

        feat = self.act(APPF * feat_select(self.basis_matrix(x), col))            # BN, k, C
        pad_x = x.unsqueeze(-1).repeat(1, 1, 1, self.k)
        return self.edge_conv(torch.cat((feat - pad_x, pad_x), dim=1)).max(dim=-1, keepdim=False)[0]   # BN, C


class seg_Net(nn.Module):
    def __init__(self, opt):
        super(seg_Net, self).__init__()
        self.k = opt.k
        num_part = 50
        self.opt = opt

        self.additional_channel = 0

        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(64*3+64*2, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(1088+64*3+64*2+3, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv6 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Conv1d(256, num_part, kernel_size=1, bias=False)

        self.conv1 = nn.Sequential(nn.Conv2d(6 + 6 + 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.gb_gconv_1 = nn.Sequential(nn.Conv2d(6, 32, (1, 1), bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(negative_slope=0.2))
        
        self.gb_gconv_2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(negative_slope=0.2))
        
        self.conv_onehot = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.pari_1 = PaRIConv(64, 64, k=opt.k)
        self.pari_2 = PaRIConv(64, 128, k=opt.k)

        self.dp1 = nn.Dropout(p=0.4)
        self.dp2 = nn.Dropout(p=0.4)
        
        self.psa = Point_Spatial_Attention(3)
        self.feature_fusion = feature_fusion(64*5,16)

    def forward(self, data, train=True):
        batch_size = data.batch.max() + 1
        BN, feat_dim = data.x.size()
        N = int(BN/batch_size)
        data.x = data.x.view(batch_size, -1, feat_dim).permute(0, 2, 1)

        euc_knn_idx = knn(data.pos.view(batch_size, -1, 3).permute(0, 2, 1), k=self.k).cuda()

        x = data.x # [32, 3, 2048]
        num_points = x.size(2)
        points0 = data.pos.view(batch_size, 3, -1)

        eq = global_transform(data.pos.view(batch_size, 3, -1), 32, train=train) # [32, 3, 2048]
        global_f = get_neighbor(self.psa(eq)) # [32, 6, 2048, 40]
        g_out1 = self.gb_gconv_1(global_f) # [32, 64, 2048, 40]
        g_out2 = self.gb_gconv_2(g_out1)  # [32, 64, 2048, 40]

        APPF, (row, col) = self.get_graph_feature(data.pos, data, idx=euc_knn_idx) # [655360, 8]
        APPF = APPF.view(batch_size, N, self.k, -1).permute(0, 3, 1, 2).contiguous() # [32, 8, 2048, 40]
        pad_x = x.unsqueeze(-1).repeat(1, 1, 1, self.k) # [32, 3, 2048, 40]
        x = self.conv1(torch.cat([APPF, feat_select(x, col) - pad_x , pad_x], dim=1))  # EdgeConv [32, 64, 2048, 40]
        x1 = x.max(dim=-1, keepdim=False)[0] # [32, 64, 2048]
        x1 = self.conv2(x1)

        APPF, edge_index = self.get_graph_feature(x1, data) # [655360, 8]
        x2 = self.pari_1(x1, APPF, edge_index, bs=batch_size) # [32, 64, 2048]
        x2 = self.conv3(x2)

        APPF, edge_index = self.get_graph_feature(x2, data) # [655360, 8]
        x3 = self.pari_2(x2, APPF, edge_index, bs=batch_size) # [32, 128, 2048]
        
        x = torch.cat((x1, x2, x3, g_out2.max(dim=-1, keepdim=False)[0]), dim=1) # [32, 64*3+64*2, num_points]
        xx = self.feature_fusion(x)
        y = self.conv4(xx) # [32, 1024, num_points]
        y = y.max(dim=-1, keepdim=True)[0] # [32, 1024, 1]
        
        onehot = data.onehot.unsqueeze(2) # [32, 16, 1]
        onehot_expand = self.conv_onehot(onehot) # [32, 64, 1]
        
        z = torch.cat((y, onehot_expand), dim=1) # [32, 1088, 1]
        z = z.repeat(1, 1, num_points) # [32, 1088, 2048]
        
        x = torch.cat((z, x, points0), dim=1) # [32, 1088+64*3+64*2, 2048]
        x = self.conv5(x) # [32, 512, num_points]
        x = self.dp1(x)
        x = self.conv6(x) # [32, 256, num_points]
        x = self.dp2(x)
        x = self.conv7(x) # [32, 50, num_points]

        return x.permute(0, 2, 1).contiguous() # [32, num_points, 50]

    def get_graph_feature(self, x, data, idx=None, flag=False):
        if idx is None:
            idx = knn(x, k=self.k).cuda()  # (batch_size, num_points, k)

        batch_size = idx.size(0)
        num_points = idx.size(1)

        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base
        col = idx.view(-1)
        row = (torch.arange(num_points, device=device).view(1, -1, 1).repeat(batch_size, 1, self.k) + idx_base).view(-1)

        pos_i = data.pos[row]
        pos_j = data.pos[col]
        if flag == True:
            norm_i = data.l0[row]
            norm_j = data.l0[col]
        else:
            norm_i = data.l0[row]
            norm_j = data.l0[col]

        x_i_axis = data.l1[row]
        x_j_axis = data.l1[col]

        # generate APPFs
        # PPF
        ppf = point_pair_features(pos_i=pos_i, pos_j=pos_j,
                                  norm_i=norm_i, norm_j=norm_j)
        # \beta_r_j
        angles_i_j = get_local_frame_angles(pos_i=pos_i,
                                            pos_j=pos_j,
                                            norm_i=norm_i,
                                            k=self.k,
                                            axis=x_i_axis)
        # \beta_j_r
        angles_j_i = get_local_frame_angles(pos_i=pos_j,
                                            pos_j=pos_i,
                                            norm_i=norm_j,
                                            k=self.k,
                                            axis=x_j_axis)

        ppf[:, 1:] = torch.cos(ppf[:, 1:])
        angles_i_j = torch.cat([torch.cos(angles_i_j), torch.sin(angles_i_j)], dim=-1)
        angles_j_i = torch.cat([torch.cos(angles_j_i), torch.sin(angles_j_i)], dim=-1)
        APPF = torch.cat((angles_i_j, angles_j_i, ppf), dim=-1)

        return APPF, [row, col]

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

def get_neighbor(x, k=40, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k, remove_self_loop=False)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    return feature
    
def global_transform(points, npoints, train):
    points = points.permute(0, 2, 1)
    idx = farthest_point_sample(points, npoints)
    centroids = index_points(points, idx)   #[B, S, C] 
    U, S, V = svd(centroids)

    if train == True:
        index = torch.randint(2, (points.size(0), 1, 3)).type(torch.FloatTensor).cuda()
        V_ = V * index
        V -= 2 * V_
    else:
        key_p = centroids[:, 0, :].unsqueeze(1)
        angle = torch.matmul(key_p, V)
        index = torch.le(angle, 0).type(torch.FloatTensor).cuda()      
        V_ = V * index
        V -= 2 * V_

    xyz = torch.matmul(points, V).permute(0, 2, 1)

    return xyz

class Point_Spatial_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Point_Spatial_Attention, self).__init__()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(in_dim)
        
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=False),
                                self.bn1,
                                nn.LeakyReLU(negative_slope=0.2), 
                                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=False))
        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn4,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        feat = self.mlp(x) # [B, 128, 1024]
        proj_query = self.query_conv(feat) # [B, 16, 1024]
        proj_key = self.key_conv(feat).permute(0, 2, 1) # [B, 1024, 16]
        similarity_mat = self.softmax(torch.bmm(proj_key, proj_query)) # [B, 1024, 1024]

        proj_value = self.value_conv(feat) # [B, 3, 1024]
        out = torch.bmm(proj_value, similarity_mat.permute(0, 2, 1))
        out = self.alpha*out + x 
        return out

class feature_fusion(nn.Module):
    def __init__(self, in_dim, reduction):
        super(feature_fusion, self).__init__()

        self.conv = nn.Sequential(
          nn.Conv1d(in_dim, in_dim//reduction, kernel_size=1, bias=False), 
          nn.BatchNorm1d(in_dim//reduction),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Conv1d(in_dim//reduction, in_dim, kernel_size=1, bias=False),
        )
    def forward(self, x):
        att = self.conv(x)
        att = F.softmax(att, dim=1)
        out = x * att
        return out