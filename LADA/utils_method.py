import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.distributions as distr
from utils_data import CustomTensorDataset
import torch.utils.data as data_utils
from torchvision import transforms

class lambdanet(nn.Module):
    def __init__(self, args):
        super(lambdanet, self).__init__()
        self.input_height = args.input_height_featuremap
        self.input_width = args.input_width_featuremap
        self.input_dim = args.input_dim_featuremap
        self.concat_input_dim = 2 * self.input_dim

        self.conv1 = nn.Conv2d(self.concat_input_dim, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if args.data == 'SVHN' or args.data == 'Cifar10' or args.data == 'Cifar100':
            self.fc1 = nn.Linear(576, 120)
        else:
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3_Uni = nn.Linear(60, 3)
        self.fc3_Beta = nn.Linear(60, 1)
        self.sftmax = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x, args):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        temp_x = self.fc3_Beta(x)
        out = self.sigm(temp_x)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def mixup_two_batches(batch1, target1, batch2, target2, args):
    batch1, target1, batch2, target2 = batch1.cuda(), target1.cuda(), batch2.cuda(), target2.cuda()

    batch_size = batch1.shape[0]
    input_dim = batch1.shape[1]
    input_height = batch1.shape[2]
    input_width = batch1.shape[3]

    target1_1 = target1.unsqueeze(1)
    target2_1 = target2.unsqueeze(1)

    y_onehot1 = torch.FloatTensor(batch_size, args.nb_classes).cuda()
    y_onehot1.zero_()
    target1_oh = y_onehot1.scatter_(1, target1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, args.nb_classes).cuda()
    y_onehot2.zero_()
    target2_oh = y_onehot2.scatter_(1, target2_1, 1)

    a = np.random.beta(args.alpha, args.alpha, [batch_size, 1])
    b = np.tile(a[..., None, None], [1, input_dim, input_height, input_width])

    lam_b_torch = torch.from_numpy(b).cuda()
    batch1 = batch1 * lam_b_torch.float()
    batch2 = batch2 * (1 - lam_b_torch.float())

    c = np.tile(a, [1, args.nb_classes])
    lam_t_torch = torch.from_numpy(c).cuda()
    lam_t_torch2 = 1 - lam_t_torch
    target1_oh = target1_oh.float() * lam_t_torch.float()
    target2_oh = target2_oh.float() * lam_t_torch2.float()

    new_batch = batch1 + batch2
    new_target_oh = target1_oh + target2_oh

    return new_batch, new_target_oh, a

def mixup_two_batches_given_lambdas(batch1, target1, batch2, target2, lam, args):

    batch_size = batch1.shape[0]
    input_dim = batch1.shape[1]
    input_height = batch1.shape[2]
    input_width = batch1.shape[3]
    target1 = target1.unsqueeze(1)
    target2 = target2.unsqueeze(1)

    y_onehot1 = torch.FloatTensor(batch_size, args.nb_classes).cuda()
    y_onehot1.zero_()
    target1_oh = y_onehot1.scatter_(1, target1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, args.nb_classes).cuda()
    y_onehot2.zero_()
    target2_oh = y_onehot2.scatter_(1, target2, 1)
    lam_b_torch = lam[..., None, None].repeat(1, input_dim, input_height, input_width)
    lam_t_torch = lam.repeat(1, args.nb_classes)
    lam_t_torch2 = 1-lam_t_torch

    batch1 = batch1 * lam_b_torch.float()
    batch2 = batch2 * (1 - lam_b_torch.float())

    target1_oh = target1_oh.float() * lam_t_torch.float()
    target2_oh = target2_oh.float() * lam_t_torch2.float()

    new_batch = batch1 + batch2
    new_target_oh = target1_oh + target2_oh

    return new_batch, new_target_oh

def train_mixed(epoch, pooled_data1, pooled_target1, pooled_data2, pooled_target2, alphas, temp_pool_index1, args):
    num_batch = 1

    # (1) shuffle data
    batch_size = pooled_data1.shape[0]
    idx_shuffle = torch.randperm(batch_size)
    shuffled_pooled_data1 = pooled_data1[idx_shuffle]
    shuffled_pooled_data2 = pooled_data2[idx_shuffle]
    shuffled_pooled_target1 = pooled_target1[idx_shuffle]
    shuffled_pooled_target2 = pooled_target2[idx_shuffle]
    if args.method == 'LADA_fixed' or args.method == 'VanillaMixup':
        shuffled_alphas = torch.from_numpy(np.array([[args.alpha]]*batch_size)).float()
    if args.method == 'LADA':
        torch_alphas = torch.from_numpy(np.array(alphas))[temp_pool_index1].cuda()
        shuffled_alphas = torch_alphas[idx_shuffle]

    # (2) sample lambdas from alphas
    alphas_cpu = shuffled_alphas.cpu()
    f = distr.beta.Beta(alphas_cpu, alphas_cpu)
    lam_cpu = f.rsample()
    lam = lam_cpu.cuda()

    # (3) forward data
    args.model.train()

    if args.data == 'Cifar10':
        transform_train = transforms.Compose(
            [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    else:
        transform_train = None
    shuffled_train_dataset1 = CustomTensorDataset(tensors=(shuffled_pooled_data1, shuffled_pooled_target1), dataname=args.data, transform=transform_train)
    shuffled_train_dataset2 = CustomTensorDataset(tensors=(shuffled_pooled_data2, shuffled_pooled_target2), dataname=args.data, transform=transform_train)
    shuffled_loader1 = data_utils.DataLoader(shuffled_train_dataset1, batch_size=batch_size, shuffle=False)
    shuffled_loader2 = data_utils.DataLoader(shuffled_train_dataset2, batch_size=batch_size, shuffle=False)
    for (batch, target) in shuffled_loader1:
        assert len(shuffled_loader1) == 1
        batch = Variable(batch).cuda()
        hidden1 = args.model(batch, args)
        target1 = target.cuda()
    for (batch, target) in shuffled_loader2:
        assert len(shuffled_loader1) == 1
        batch = Variable(batch).cuda()
        hidden2 = args.model(batch, args)
        target2 = target.cuda()

    mixed_hidden, mixed_target = mixup_two_batches_given_lambdas(hidden1, target1, hidden2, target2, lam, args)
    output = args.model.forward_rest(mixed_hidden, args)
    m = nn.LogSoftmax(dim=1)
    loss = -m(output) * mixed_target
    loss = torch.sum(loss) / batch_size
    loss.backward()
    args.optimizer.step()

    train_loss = loss.item() * batch_size

    output_for_tsne = [shuffled_pooled_data1, shuffled_pooled_data2, shuffled_pooled_target1, shuffled_pooled_target2, lam]

    return train_loss, num_batch, batch_size, output_for_tsne

def get_kth_feature_map(data, args):
    data = data.cuda()
    kth_feature_map = args.model.kth_featuremap(data, args)
    return kth_feature_map

def remove_pooled_points(pool_data, pool_target, pool_subset_dropout, pool_data_dropout, pool_target_dropout, pool_index):

    np_data = pool_data.numpy()
    np_target = pool_target.numpy()

    pool_data_dropout = pool_data_dropout.numpy()
    pool_target_dropout = pool_target_dropout.numpy()

    pool_subset_idx = pool_subset_dropout.numpy()
    np_index = pool_index.numpy()

    np_data = np.delete(np_data, pool_subset_idx, axis=0)
    np_target = np.delete(np_target, pool_subset_idx, axis=0)

    pool_data_dropout = np.delete(pool_data_dropout, np_index, axis=0)
    pool_target_dropout = np.delete(pool_target_dropout, np_index, axis=0)

    np_data = np.concatenate((np_data, pool_data_dropout), axis=0)
    np_target = np.concatenate((np_target, pool_target_dropout), axis=0)
    pool_data = torch.from_numpy(np_data)
    pool_target = torch.from_numpy(np_target)

    return pool_data, pool_target

def evaluate_featuremap(featuremap, args):
    args.model.eval()

    predictions = []
    featuremap = featuremap.cuda()
    output = args.model.forward_rest(featuremap, args)
    softmaxed = F.softmax(output, dim=1)
    predictions.append(softmaxed)
    predictions = torch.cat(predictions, dim=0)

    return predictions

def print_total_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    elapsed_hours = int(elapsed_mins / 60)
    elapsed_mins = int(elapsed_mins - (elapsed_hours * 60))

    if elapsed_hours == 0:
        print('...%dmin %dsec'%(elapsed_mins, elapsed_secs))
    if elapsed_hours != 0:
        print('...%dhour %dmin %dsec'%(elapsed_hours, elapsed_mins, elapsed_secs))

def print_memory_usage():
    t = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    c = torch.cuda.memory_cached(0) / (1024 ** 2)
    a = torch.cuda.memory_allocated(0) / (1024 ** 2)
    f = c - a
    print(t, c, a, f)

def print_number_of_data(args):
    print('...Train data : %d' % len(args.train_data))
    print('...Pool data : %d' % len(args.pool_data))
    print('...Test data : %d' % len(args.test_loader.dataset))

