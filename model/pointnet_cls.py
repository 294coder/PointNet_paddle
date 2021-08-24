import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import pdb

paddle.disable_static()
paddle.set_device('cpu')

class STN3d(nn.Layer):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2)
        # pdb.set_trace()
        # x = x.reshape(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = paddle.to_tensor(np.eye(3).flatten()).reshape((1, 9)).expand((batchsize, -1))
        x = x + iden
        x = x.reshape((-1, 3, 3))
        return x


class STNkd(nn.Layer):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=False)
        x = x.reshape([-1, 1024])

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = paddle.to_tensor(np.eye(self.k).flatten()).reshape([1, self.k * self.k]).expand([batchsize, -1])
        x = x + iden
        x = x.reshape([-1, self.k, self.k])
        return x


class PointNetEncoder(nn.Layer):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.shape
        trans = self.stn(x)
        x = x.transpose((0, 2, 1))
        if D > 3:
            x, feature = x.split(3, axis=2)
        x = paddle.bmm(x, trans)
        if D > 3:
            x = paddle.concat([x, feature], axis=2)
        x = x.transpose((0, 2, 1))
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose((0, 2, 1))
            x = paddle.bmm(x, trans_feat)
            x = x.transpose((0, 2, 1))
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = paddle.max(x, 2)
        x = x.reshape([-1, 1024])
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.reshape(-1, 1024, 1).expand(-1, -1, N)
            return paddle.concat([x, pointfeat], 1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.shape[1]
    batchsize = trans.shape[0]
    I = paddle.eye(d).unsqueeze(0)
    loss = paddle.mean(paddle.norm(paddle.bmm(trans, trans.transpose((0, 2, 1))) - I, axis=(1, 2)))
    return loss


class get_model(nn.Layer):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, axis=1)
        return x, trans_feat


class get_loss(nn.Layer):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target.astype(paddle.int64))
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


