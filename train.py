import os
import pdb

import paddle
from model.pointnet_cls import get_model, get_loss
from utils.modelnet_dataloader import ModelNetDataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

epochs = 50
val_ep = 5

module = get_model()
loss = get_loss()
train_datasets = ModelNetDataLoader(r'.\data\modelnet40_normal_resampled',
                                    normal_channel=False)
val_datasets = ModelNetDataLoader(r'.\data\modelnet40_normal_resampled', split='test',
                                  normal_channel=False)

train_dataloader = paddle.io.DataLoader(train_datasets, batch_size=16, shuffle=True)
val_dataloader = paddle.io.DataLoader(val_datasets, batch_size=16, shuffle=False)
opt = paddle.optimizer.AdamW(parameters=module.parameters(), learning_rate=1e-3)

for ep in range(epochs):
    print(f'epoch: {ep + 1}')
    train_loss = 0.
    train_acc = 0.
    pbar = tqdm(train_dataloader)
    module.train()
    for X, y in pbar:
        # y = y.reshape([40, -1])
        pred, trans_feat = module(X)
        # loss
        l = loss(pred, y, trans_feat)
        train_loss += l.numpy()
        l.backward()
        # acc
        acc = accuracy_score(pred.argmax(1).numpy(), y.reshape([-1]).numpy())
        train_acc += acc
        # pdb.set_trace()
        pbar.set_description_str(f'loss:{l.item():.4f}, acc:{acc:.3f}')
        # optimizer
        opt.step()
        opt.clear_grad()
    print(
        '----train---' + f'mean loss: {train_loss / len(train_dataloader)}, mean_acc: {train_acc / len(train_dataloader)}')
    if ep % val_ep == 0:
        pbar_v = tqdm(val_dataloader)
        val_loss = 0.
        val_acc = 0.
        module.eval()
        with paddle.set_grad_enabled(False):
            for X, y in pbar_v:
                pred, trans_feat = module(X)
                l = loss(pred, y, trans_feat)
                val_loss += l.numpy()
                acc = accuracy_score(pred.argmax(1).numpy(), y.reshape([-1]).numpy())
                pbar.set_description(f'loss:{l.item():.3f}, acc:{acc:.3f}')
            print(
                '----eval---' + f'mean loss: {train_loss / len(train_dataloader)}, mean_acc: {train_acc / len(train_dataloader)}')


