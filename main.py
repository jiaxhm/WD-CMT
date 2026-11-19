import os
import glob
import time
import math
import copy
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
from utils.utils import Logger, mkdirs
from models.wd_cmt import WDCMT


class DefaultConfigs(object):
    seed = 345 # 345#666
    # 优化器及训练参数
    weight_decay = 5e-4
    momentum = 0.9
    init_lr = 0.001
    train_epoch = 100
    test_epoch = 1
    BATCH_SIZE_TRAIN = 64
    gpus = '0'
    # 目标数据集 (切换测试: 'Houston2013' / 'KSC' / 'longkou' / 'hanchuan')
    data = 'longkou'
    # 参数
    patch_size = 11
    pca_components = 30
    model_type = 'WD-CMT'
    depth = 1
    embed_dim = 32
    dim_inner = 2 * embed_dim
    dt_rank = math.ceil(embed_dim / 16)
    d_state = 16
    group_type = 'Cube'
    # 连续三维蛇形：光谱优先、空间优先及各自反向；相邻步骤不跨行或跨层跳跃。
    scan_type = 'Serpentine spectral-spatial'
    k_group = 4
    pos = False
    cls = False
    conv3D_channel = 32
    conv3D_kernel = (3, 5, 5)

    @property
    def dim_patch(self):
        return self.patch_size - self.conv3D_kernel[1] + 1

    @property
    def dim_linear(self):
        return self.pca_components - self.conv3D_kernel[0] + 1

    checkpoint_path = './checkpoint_merged/'


config = DefaultConfigs()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 动态更新 checkpoint_path，为每个数据集创建独立的子文件夹 ---
config.checkpoint_path = os.path.join(config.checkpoint_path, config.data)
os.makedirs(config.checkpoint_path, exist_ok=True)


def loadData():
    if config.data == 'Houston2013':
        data = sio.loadmat('./data/Houston2013/Houston.mat')['Houston']
        labels = sio.loadmat('./data/Houston2013/Houston_gt.mat')['Houston_GT']
    elif config.data == 'KSC':
        data = sio.loadmat('data/KSC/KSC_corrected.mat')['KSC']
        labels = sio.loadmat('data/KSC/KSC_gt.mat')['KSC_gt']
    elif config.data == 'longkou':
        data = sio.loadmat('data/longkou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        labels = sio.loadmat('data/longkou/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
    elif config.data == 'hanchuan':
        data = sio.loadmat('data/hanchuan/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
        labels = sio.loadmat('data/hanchuan/WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
    else:
        raise ValueError(f"Unknown dataset: {config.data}")
    return data, labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def createImageCubes(X, y, windowSize=5):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    valid_pixels = np.count_nonzero(y > 0)
    patchesData = np.zeros((valid_pixels, windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros(valid_pixels)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] == 0:
                continue
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin] - 1  # 标签归零化
            patchIndex += 1
    return patchesData, patchesLabels


class MemoryEfficientDS(Dataset):
    def __init__(self, full_image, indices, labels, patch_size):
        self.full_image = full_image
        self.indices = indices
        self.labels = labels
        self.patch_size = patch_size
        margin = patch_size // 2
        self.padded_image = np.pad(full_image, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        w = self.full_image.shape[1]
        r = idx // w
        c = idx % w
        patch = self.padded_image[r: r + self.patch_size, c: c + self.patch_size, :]
        patch = torch.from_numpy(patch).float()
        patch = patch.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        target = torch.from_numpy(np.array(self.labels[i])).long()
        return patch, target


def create_data_loader():
    X, y = loadData()
    groundtruth = y.copy()
    y_flat_full = y.reshape(-1)
    all_valid_indices = np.nonzero(y_flat_full)[0]
    all_valid_labels = y_flat_full[all_valid_indices] - 1
    num_classes = int(np.max(y))

    print(f'\n--- 开始处理数据集: {config.data} (类别总数: {num_classes}) ---')
    X_pca = applyPCA(X, numComponents=config.pca_components)

    # 提取标签数据用于统计
    train_labels_stat = []
    val_labels_stat = []
    test_labels_stat = []

    if config.data in ['longkou', 'hanchuan']:
        train_idx_inner, rest_idx_inner = [], []
        np.random.seed(config.seed)
        for cls in np.unique(all_valid_labels):
            c_idx = np.where(all_valid_labels == cls)[0]
            np.random.shuffle(c_idx)
            train_idx_inner.extend(c_idx[:50])
            rest_idx_inner.extend(c_idx[50:])
        val_idx_inner, test_idx_inner = train_test_split(
            rest_idx_inner, test_size=0.90, random_state=config.seed, stratify=all_valid_labels[rest_idx_inner]
        )

        train_labels_stat = all_valid_labels[train_idx_inner]
        val_labels_stat = all_valid_labels[val_idx_inner]
        test_labels_stat = all_valid_labels[test_idx_inner]

        train_set = MemoryEfficientDS(X_pca, all_valid_indices[train_idx_inner], train_labels_stat, config.patch_size)
        val_set = MemoryEfficientDS(X_pca, all_valid_indices[val_idx_inner], val_labels_stat, config.patch_size)
        test_set = MemoryEfficientDS(X_pca, all_valid_indices[test_idx_inner], test_labels_stat, config.patch_size)
        all_set = MemoryEfficientDS(X_pca, all_valid_indices, all_valid_labels, config.patch_size)

    elif config.data in ['Houston2013', 'KSC']:
        X_cubes, y_cubes = createImageCubes(X_pca, groundtruth, windowSize=config.patch_size)
        X_cubes = X_cubes.reshape(-1, config.patch_size, config.patch_size, config.pca_components, 1).transpose(0, 4, 3,
                                                                                                                1, 2)
        indices = np.arange(len(y_cubes))
        train_ratio = 0.10 if config.data == 'Houston2013' else 0.05
        train_idx, rest_idx = train_test_split(indices, train_size=train_ratio, random_state=config.seed,
                                               stratify=y_cubes)
        val_idx, test_idx = train_test_split(rest_idx, test_size=0.90, random_state=config.seed,
                                             stratify=y_cubes[rest_idx])

        train_labels_stat = y_cubes[train_idx]
        val_labels_stat = y_cubes[val_idx]
        test_labels_stat = y_cubes[test_idx]

        train_set = TensorDataset(torch.FloatTensor(X_cubes[train_idx]), torch.LongTensor(train_labels_stat))
        val_set = TensorDataset(torch.FloatTensor(X_cubes[val_idx]), torch.LongTensor(val_labels_stat))
        test_set = TensorDataset(torch.FloatTensor(X_cubes[test_idx]), torch.LongTensor(test_labels_stat))
        all_set = TensorDataset(torch.FloatTensor(X_cubes), torch.LongTensor(y_cubes))

    # --- 生成样本数量统计文本块 (准备写入文件，不在控制台 print) ---
    sample_report = f"\n{'-' * 60}\n"
    sample_report += f"📊 数据集 {config.data} 样本分布统计 (Seed: {config.seed})\n"
    sample_report += f"{'-' * 60}\n"
    sample_report += f"{'Class':<8} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}\n"
    sample_report += f"{'-' * 60}\n"

    total_train, total_val, total_test = len(train_labels_stat), len(val_labels_stat), len(test_labels_stat)

    for c in range(num_classes):
        c_train = np.sum(train_labels_stat == c)
        c_val = np.sum(val_labels_stat == c)
        c_test = np.sum(test_labels_stat == c)
        c_total = c_train + c_val + c_test
        sample_report += f"Class {c + 1:<2} | {c_train:<8} | {c_val:<8} | {c_test:<8} | {c_total:<8}\n"

    sample_report += f"{'-' * 60}\n"
    sample_report += f"{'TOTAL':<8} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {total_train + total_val + total_test:<8}\n"
    sample_report += f"{'-' * 60}\n"

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False)
    all_loader = DataLoader(all_set, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False)

    return train_loader, val_loader, test_loader, all_valid_indices, all_loader, groundtruth, num_classes, sample_report


def fine_grained_hsic_loss(feat1, feat2):
    # 【修复 1】在此处将来自网络底层的 (B, T, H, W, Channel)
    # 转换回标准的 (B, Channel, T, H, W) 供后续计算
    feat1 = feat1.permute(0, 4, 1, 2, 3).contiguous()
    feat2 = feat2.permute(0, 4, 1, 2, 3).contiguous()

    B, C, T, H, W = feat1.shape
    L = T * H * W
    z1 = F.normalize(feat1.view(B, C, L), p=2, dim=2)
    z2 = F.normalize(feat2.view(B, C, L), p=2, dim=2)
    K = torch.bmm(z1, z1.transpose(1, 2))
    L_mat = torch.bmm(z2, z2.transpose(1, 2))
    H_mat = torch.eye(C, device=feat1.device) - (1.0 / C)
    H_mat = H_mat.unsqueeze(0).expand(B, -1, -1)
    Kc = torch.bmm(torch.bmm(H_mat, K), H_mat)
    hsic = torch.sum(Kc * L_mat, dim=(1, 2)) / ((C - 1) ** 2)
    return hsic.mean()


def acc_reports(y_test, y_pred_test):
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc = np.nan_to_num(truediv(np.diag(confusion), np.sum(confusion, axis=1)))
    aa = np.mean(each_acc)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def mytest(net, loader):
    net.eval()
    preds, targets = [], []
    start_time = time.time()  # 记录推理开始时间
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs, _, _, _ = net(inputs)
            preds.append(np.argmax(outputs.detach().cpu().numpy(), axis=1))
            targets.append(labels.numpy())
    infer_time = time.time() - start_time  # 计算推理耗时
    return np.concatenate(preds), np.concatenate(targets), infer_time


def train(train_loader, val_loader, num_classes):
    net = WDCMT(
        group_type=config.group_type, k_group=config.k_group,
        embed_dim=config.embed_dim, dt_rank=config.dt_rank,
        d_inner=config.dim_inner, d_state=config.d_state,
        num_classes=num_classes, depth=config.depth,
        scan_type=config.scan_type, pos=config.pos, cls=config.cls,
        conv3D_channel=config.conv3D_channel, conv3D_kernel=config.conv3D_kernel,
        dim_patch=config.dim_patch, dim_linear=config.dim_linear,
    ).to(device)
    # Houston2013 的第一批数据到达小波分支时，自动输出 1D/2D 小波响应图。
    if config.data == 'Houston2013':
        net.enable_wavelet_visualization(output_dir='new photo')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epoch, eta_min=1e-5)

    best_val_oa = 0.0
    best_model_in_mem = None
    test_interval = 10

    start_train_time = time.time()  # 记录训练开始时间

    for epoch in range(config.train_epoch):
        net.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs, _, feat_mamba, feat_trans = net(data)
            loss_cls = criterion(outputs, target)
            loss_hsic = fine_grained_hsic_loss(feat_mamba, feat_trans)
            loss = loss_cls + 0.8 * loss_hsic
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % test_interval == 0 or (epoch + 1) == config.train_epoch:
            y_pred_val, y_val, _ = mytest(net, val_loader)
            val_oa, _, _, val_aa, val_kappa = acc_reports(y_val, y_pred_val)
            print(
                f"[Epoch: {epoch + 1:03d}] Loss: {avg_loss:.4f} | Val_OA: {val_oa:.2f}% | Val_AA: {val_aa:.2f}% | Val_Kappa: {val_kappa:.2f}%")
            if val_oa > best_val_oa:
                best_val_oa = val_oa
                best_model_in_mem = copy.deepcopy(net.state_dict())
                print(f"最佳验证分数更新 -> Val_OA: {best_val_oa:.2f}%")
        else:
            print(f"[Epoch: {epoch + 1:03d}] Loss: {avg_loss:.4f}")
    total_train_time = time.time() - start_train_time
    if best_model_in_mem is not None:
        net.load_state_dict(best_model_in_mem)
        print("\n提取最强验证集权重，准备全量测试...")
    return net, total_train_time

if __name__ == '__main__':
    if config.data in ['longkou', 'hanchuan']:
        train_scale_str = '50'
    elif config.data == 'Houston2013':
        train_scale_str = '0.10'
    elif config.data == 'KSC':
        train_scale_str = '0.05'
    else:
        train_scale_str = 'custom'
    log = Logger()
    log.open(os.path.join(config.checkpoint_path, f'result-{train_scale_str}.txt'), mode='a')
    global best_test_oa, best_model_path
    best_test_oa = 0.0
    best_model_path = ""

    for num in range(config.test_epoch):
        train_loader, val_loader, test_loader, index, all_loader, groundtruth, num_classes, sample_report = create_data_loader()
        log.write(sample_report + "\n")
        net, train_time = train(train_loader, val_loader, num_classes)
        try:
            from thop import profile

            dummy_input = torch.randn(1, 1, config.pca_components, config.patch_size, config.patch_size).to(device)
            macs, params = profile(net, inputs=(dummy_input,), verbose=False)
            flops = macs * 2
            flops_str = f"{flops / 1e9:.4f} G"
            params_str = f"{params / 1e6:.4f} M"
        except Exception as e:
            flops_str = "N/A (建议安装 thop: pip install thop)"
            params_str = "N/A"
        print("\n" + "=" * 60)
        print(f"🏆 第 {num + 1} 组实验：测试集")
        print("=" * 60)
        y_pred_test, y_test, infer_time = mytest(net, test_loader)
        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
        test_report = (
            f"--- 实验 Run {num + 1} 测试集最终结果 ---\n"
            f"OA: {each_oa:.2f}%\n"
            f"AA: {each_aa:.2f}%\n"
            f"Kappa: {each_kappa:.2f}%\n"
            f"训练时间 (Train Time): {train_time:.2f} s\n"
            f"推理时间 (Inference Time): {infer_time:.2f} s\n"
            f"计算量 (FLOPs): {flops_str}\n"
            f"总参数量 (Params): {params_str}\n"
            f"各类别细分准确率:\n"
        )
        for i, a in enumerate(each_acc):
            test_report += f"  Class {i + 1:02d}: {a:.2f}%\n"
        print(test_report)
        log.write(test_report + "\n")
        if each_oa > best_test_oa:
            best_test_oa = each_oa
            best_name = f'best-{train_scale_str}.pth'
            new_save_path = os.path.join(config.checkpoint_path, best_name)
            torch.save(net.state_dict(), new_save_path)
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = new_save_path
