import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.distributions import uniform
from model_ResNet import ResNet18
from model_SharedResNet import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data as data_utils
import torchvision
from torchvision import datasets, transforms


def make_loader(args, kwargs):

    if args.data == 'Fashion':
        train_loader_all = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader_all, test_loader, None, None

    if args.data == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        trainset_all = torchvision.datasets.SVHN(
            root='data', split='train', download=True, transform=transform_train)
        train_loader_all = torch.utils.data.DataLoader(
            trainset_all, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        testset = torchvision.datasets.SVHN(
            root='data', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        return train_loader_all, test_loader, trainset_all, testset

    if args.data == 'Cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset_all = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform_train)
        train_loader_all = torch.utils.data.DataLoader(
            trainset_all, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        return train_loader_all, test_loader, trainset_all, testset

    if args.data == 'Cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset_all = torchvision.datasets.CIFAR100(
            root='data', train=True, download=True, transform=transform_train)
        train_loader_all = torch.utils.data.DataLoader(
            trainset_all, batch_size=args.batch_size, shuffle=True, **kwargs)

        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset = torchvision.datasets.CIFAR100(
            root='data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        return train_loader_all, test_loader, trainset_all, testset

    else:
        raise ValueError

def prepare_data(train_loader_all, test_loader, trainset_all, testset, args):

    if args.data == 'Fashion':

        train_data_all = train_loader_all.dataset.train_data
        train_target_all = train_loader_all.dataset.train_labels
        shuffler_idx = torch.randperm(train_target_all.size(0))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = test_loader.dataset.test_data
        test_target = test_loader.dataset.test_labels

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all.unsqueeze_(1)
        test_data.unsqueeze_(1)

        train_data_all = train_data_all.float()
        test_data = test_data.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:2], :, :, :]
            target_i = train_target_all.numpy()[idx[0][0:2]]
            pool_data_i = train_data_all.numpy()[idx[0][2:], :, :, :]
            pool_target_i = train_target_all.numpy()[idx[0][2:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")  # 10 x 10 x 1 x 28 x 28 --> 100 x 1 x 28 x 28
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")  # 10 x 10 x 1 x 28 x 28 --> 100 x 1 x 28 x 28
        pool_target = np.concatenate(pool_target, axis=0)

        mean = torch.tensor([0.1307])
        std = torch.tensor([0.3081])
        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = (test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

    if args.data == 'SVHN':

        train_data_all = torch.Tensor(trainset_all.data)
        train_target_all = torch.Tensor(trainset_all.labels).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset.data)
        test_target = torch.Tensor(testset.labels).long()
        test_data = test_data.float().numpy()

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:2], :, :, :]
            target_i = train_target_all.numpy()[idx[0][0:2]]
            pool_data_i = train_data_all.numpy()[idx[0][2:], :, :, :]
            pool_target_i = train_target_all.numpy()[idx[0][2:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

    if args.data == 'Cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        train_data_all = torch.Tensor(trainset_all.data)
        train_target_all = torch.Tensor(trainset_all.targets).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset.data)
        test_target = torch.Tensor(testset.targets).long()
        test_data = test_data.float().numpy()
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:2], :, :,:]
            data_i = np.transpose(data_i, (0, 3, 1, 2))
            target_i = train_target_all.numpy()[idx[0][0:2]]
            pool_data_i = train_data_all.numpy()[idx[0][2:], :, :, :]
            pool_data_i = np.transpose(pool_data_i, (0, 3, 1, 2))
            pool_target_i = train_target_all.numpy()[idx[0][2:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

    if args.data == 'Cifar100':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        train_data_all = torch.Tensor(trainset_all.data)
        train_target_all = torch.Tensor(trainset_all.targets).long()
        shuffler_idx = torch.randperm(len(train_target_all))
        train_data_all = train_data_all[shuffler_idx]
        train_target_all = train_target_all[shuffler_idx]

        test_data = torch.Tensor(testset.data)
        test_target = torch.Tensor(testset.targets).long()
        test_data = test_data.float().numpy()
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = []
        train_target = []
        pool_data = []
        pool_target = []

        train_data_all = train_data_all.float()

        for i in range(0, args.nb_classes):
            arr = np.array(np.where(train_target_all.numpy() == i))
            idx = np.random.permutation(arr)
            data_i = train_data_all.numpy()[idx[0][0:10], :, :,:]
            data_i = np.transpose(data_i, (0, 3, 1, 2))
            target_i = train_target_all.numpy()[idx[0][0:10]]
            pool_data_i = train_data_all.numpy()[idx[0][10:], :, :, :]
            pool_data_i = np.transpose(pool_data_i, (0, 3, 1, 2))
            pool_target_i = train_target_all.numpy()[idx[0][10:]]
            train_data.append(data_i)
            train_target.append(target_i)
            pool_data.append(pool_data_i)
            pool_target.append(pool_target_i)
            # print(len(data_i))
            # print(len(train_data))

        train_data = np.concatenate(train_data, axis=0).astype("float32")
        train_target = np.concatenate(train_target, axis=0)
        pool_data = np.concatenate(pool_data, axis=0).astype("float32")
        pool_target = np.concatenate(pool_target, axis=0)

        train_data_final = torch.from_numpy(train_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        train_target_final = torch.from_numpy(train_target)
        pool_data_final = torch.from_numpy(pool_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        pool_target_final = torch.from_numpy(pool_target)
        test_data_final = torch.from_numpy(test_data / 255).float().sub(mean[None, :, None, None]).div(std[None, :, None, None])
        test_target_final = test_target

        return train_data_final, train_target_final, pool_data_final, pool_target_final, test_data_final, test_target_final

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, dataname, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.dataname = dataname
        self.transform = transform

    def __getitem__(self, index):
        if self.dataname == 'Cifar10' or self.dataname == 'Cifar100':
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            x = self.tensors[0][index] * std[:, None, None] + mean[:, None, None]
        elif self.dataname == 'Fashion' or self.dataname == 'SVHN':
            x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]


def initialize_train_set(train_data, train_target, test_data, test_target, args):
    if args.data == 'Cifar10' or args.data == 'Cifar100':
        transform_train = transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
        transform_train = None
    train_dataset = CustomTensorDataset(tensors=(train_data, train_target), dataname=args.data, transform=transform_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test = data_utils.TensorDataset(test_data, test_target)
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False)
    return train_loader, test_loader

class lenet(nn.Module):
    def __init__(self, args):
        super(lenet, self).__init__()
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.input_dim = args.input_dim
        self.class_num = args.nb_classes

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class lenetMultiChannel(nn.Module):
    def __init__(self, args):
        super(lenetMultiChannel, self).__init__()
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.input_dim = args.input_dim
        self.class_num = args.nb_classes

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def init_model(args, first=False):
    if first:
        if args.method == 'AdaMixup':
            args.model = SharedResNet18(args).cuda()
            args.policy_generator = PolicyGenerator(args).cuda()
            args.intrusion_discriminator = IntrusionDiscriminator(args).cuda()
            if args.isInit == 'True':
                print('...Save model')
                torch.save(args.model.state_dict(), args.modelDir + '/model_Ada_init.pt')
                torch.save(args.policy_generator.state_dict(), args.modelDir + '/generator_Ada_init.pt')
                torch.save(args.intrusion_discriminator.state_dict(), args.modelDir + '/discriminator_Ada_init.pt')
            args.params = list(args.model.parameters()) + list(args.policy_generator.parameters()) + list(args.intrusion_discriminator.parameters())
            args.optimizer = optim.Adam(args.params, lr=args.lr, betas=(args.beta1, args.beta2))
            args.criterion = nn.CrossEntropyLoss()
            args.intrusion_criterion = nn.CrossEntropyLoss()
            args.sampler = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        else:
            args.model = ResNet18(args).cuda()
            if args.isInit == 'True':
                print('...Save model')
                torch.save(args.model.state_dict(), args.modelDir + '/model_init.pt')
            args.optimizer = optim.Adam(args.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            args.criterion = nn.CrossEntropyLoss()
        args.best_acc = 0.
    else:
        if args.method == 'AdaMixup':
            args.model = SharedResNet18(args).cuda()
            args.policy_generator = PolicyGenerator(args).cuda()
            args.intrusion_discriminator = IntrusionDiscriminator(args).cuda()
            if args.isInit == 'True':
                print('...Load model')
                args.model.load_state_dict(torch.load(args.modelDir + '/model_Ada_init.pt'))
                args.policy_generator.load_state_dict(torch.load(args.modelDir + '/generator_Ada_init.pt'))
                args.intrusion_discriminator.load_state_dict(torch.load(args.modelDir + '/discriminator_Ada_init.pt'))
            args.params = list(args.model.parameters()) + list(args.policy_generator.parameters()) + list(args.intrusion_discriminator.parameters())
            args.optimizer = optim.Adam(args.params, lr=args.lr, betas=(args.beta1, args.beta2))
            args.criterion = nn.CrossEntropyLoss()
            args.intrusion_criterion = nn.CrossEntropyLoss()
            args.sampler = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        else:
            args.model = ResNet18(args).cuda()
            if args.isInit == 'True':
                print('...Load model')
                args.model.load_state_dict(torch.load(args.modelDir + '/model_init.pt'))
            args.optimizer = optim.Adam(args.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            args.criterion = nn.CrossEntropyLoss()

def train(epoch, args):

    args.model.train()
    num_batch = 0
    num_data = 0
    total_train_loss = 0.0

    for batch_idx, (batch, target) in enumerate(args.train_loader):
        num_batch += 1
        batch_size = batch.shape[0]
        num_data += batch_size
        batch, target = batch.cuda(), target.cuda()
        batch, target = Variable(batch), Variable(target)

        args.optimizer.zero_grad()
        if args.method == 'AdaMixup':
            bef, output = args.model(batch)
        else:
            output = args.model(batch, args)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)
        loss.backward()
        args.optimizer.step()
        total_train_loss += loss.item() * batch_size

    return total_train_loss, num_batch, num_data


def evaluate2(data_type, loader, data, args):

    predictions = []

    if data_type == 'loader':
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if args.method == 'AdaMixup':
                _, output = args.model(data)
            else:
                output = args.model(data, args)
            softmaxed = F.softmax(output, dim=1)
            predictions.append(softmaxed)
        predictions = torch.cat(predictions, dim=0)

    if data_type == 'batch':
        args.model.eval()
        data = data.cuda()
        if args.method == 'AdaMixup':
            _, output = args.model(data)
        else:
            output = args.model(data, args)
        softmaxed = F.softmax(output, dim=1)
        predictions.append(softmaxed)
        predictions = torch.cat(predictions, dim=0)

    return predictions


def evaluate(loader, args, stochastic=False, predict_classes=False):
    if stochastic:
        args.model.train()
    else:
        args.model.eval()

    predictions = []
    test_loss = 0
    correct = 0

    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if args.method == 'AdaMixup':
            _, output = args.model(data)
        else:
            output = args.model(data, args)
        softmaxed = F.softmax(output.cpu(), dim=1)

        if predict_classes:
            predictions.extend(np.argmax(softmaxed.data.numpy(), axis=-1))
        else:
            predictions.extend(softmaxed.data.numpy())

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        test_loss += loss.item()
        pred = output.data.max(1)[1]
        pred = pred.eq(target.data).cpu().data.float()
        correct += pred.sum()

    return test_loss, correct, predictions

def test(epoch, args):

    test_loss, correct, _ = evaluate(args.test_loader, args, stochastic=False)

    test_loss /= len(args.test_loader)
    test_acc = 100. * correct / len(args.test_loader.dataset)

    if test_acc > args.best_acc:
        args.best_acc = test_acc


    return test_loss, test_acc, args.best_acc

