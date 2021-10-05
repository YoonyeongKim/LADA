import time
import os
from scipy.stats import mode
import copy
import torch.utils.data as data_utils

from utils_data import *
from method_AcqOnly import *
from method_AcqThenVanillaManifold import *
from method_InfoManifoldAcqBeta import *
from method_VanillaManifoldAcq import *
from method_Random import *
from method_AcqThenAdaMixup import *


def getAcquisitionFunction(name):
    if name == "MAX_ENTROPY":
        return max_entroy_acquisition
    else:
        print("ACQUSITION FUNCTION NOT IMPLEMENTED")
        raise ValueError

def acquire_points(argument, train_data, train_target, pool_data, pool_target, args, random_sample=False):

    args.pool_all = np.zeros(shape=(1))

    args.acquisition_function = getAcquisitionFunction(args.acqType)

    args.test_acc_hist = []
    args.train_loss_hist = []

    for i in range(args.ai):
        st = time.time()
        args.pool_subset = 2000
        print('---------------------------------')
        print("Acquisition Iteration " + str(i+1))
        print_number_of_data(args)

        print('(step1) Choose useful data')
        if args.method == 'LADA':
            pooled_data, pooled_target, pooled_target_oh, pooled_data1, pooled_target1, pooled_data2, pooled_target2, outs, temp_pool_index1, layer_mix = acquire_InfoManifoldMixupAcq_Beta(
                args)

            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            args.train_loader = train_loader

            total_train_loss = 0.0
            if args.isInit == 'True':
                init_model(args, first=False)
            for epoch in range(args.epochs2):
                # (1) labeled + pooled
                args.layer_mix = None
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
                # (2) mixup
                args.layer_mix = layer_mix
                train_loss_mixed, num_batch_mixed, num_data_mixed, output_for_tsne = train_mixed(epoch, pooled_data1, pooled_target1, pooled_data2, pooled_target2, outs, temp_pool_index1, args)
                # (3) sum up the loss
                total_train_loss += (train_loss_labeled + train_loss_mixed) / (num_data_labeled + num_data_mixed)

            total_train_loss /= args.epochs2

        if args.method == 'LADA_fixed':
            pooled_data, pooled_target, pooled_target_oh, pooled_data1, pooled_target1, pooled_data2, pooled_target2, layer_mix = acquire_VanillaManifoldMixupAcq(
                args)

            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            args.train_loader = train_loader

            total_train_loss = 0.0
            if args.isInit == 'True':
                init_model(args, first=False)
            for epoch in range(args.epochs2):
                # (1) labeled + pooled
                args.layer_mix = None
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
                # (2) mixup
                args.layer_mix = layer_mix
                train_loss_mixed, num_batch_mixed, num_data_mixed, output_for_tsne = train_mixed(epoch, pooled_data1, pooled_target1, pooled_data2, pooled_target2, None, None, args)
                # (3) sum up the loss
                total_train_loss += (train_loss_labeled + train_loss_mixed) / (num_data_labeled + num_data_mixed)

            total_train_loss /= args.epochs2

        if args.method == 'VanillaMixup':
            pooled_data, pooled_target, pooled_target_oh = acquire_AcqThenVanillaManifoldMixup(args)

            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            args.train_loader = train_loader

            total_train_loss = 0.0

            if args.isInit == 'True':
                init_model(args, first=False)

            args.layer_mix = random.randint(args.rangeLayerMixFrom, args.rangeLayerMixTo)
            layer_mix = args.layer_mix
            for epoch in range(args.epochs2):
                # (1) labeled + pooled
                args.layer_mix = None
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
                # (2) mixup
                args.layer_mix = layer_mix
                mixup_index = np.array(range(0, args.numQ))
                np.random.shuffle(mixup_index)
                mixup1_index = mixup_index[:int(args.numQ / 2)]
                mixup1_index = torch.from_numpy(mixup1_index).long()
                mixup1_data = pooled_data[mixup1_index]
                mixup1_target = pooled_target[mixup1_index]
                mixup2_index = mixup_index[int(args.numQ / 2):]
                mixup2_index = torch.from_numpy(mixup2_index).long()
                mixup2_data = pooled_data[mixup2_index]
                mixup2_target = pooled_target[mixup2_index]
                train_loss_mixed, num_batch_mixed, num_data_mixed, output_for_tsne = train_mixed(epoch, mixup1_data, mixup1_target, mixup2_data, mixup2_target, None, None, args)
                # (3) sum up the loss
                total_train_loss += (train_loss_labeled + train_loss_mixed) / (num_data_labeled + num_data_mixed)

            total_train_loss /= args.epochs2

        if args.method == 'AdaMixup':
            pooled_data, pooled_target, pooled_target_oh = acquire_AcqOnly(args)

            # (1) For Training with Acquired Data
            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            # cat_data = data_utils.TensorDataset(args.train_data, args.train_target)
            # train_loader = data_utils.DataLoader(cat_data, batch_size=args.batch_size, shuffle=True)
            args.train_loader = train_loader

            # (2) For Training with AdaMixup Data
            if args.data == 'Cifar10' or args.data == 'Cifar100':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            temp_pooled_dataset = CustomTensorDataset(tensors=(pooled_data, pooled_target), dataname=args.data, transform=transform_train)
            pooled_loader = data_utils.DataLoader(temp_pooled_dataset, batch_size=args.batch_size, shuffle=True)
            args.pooled_loader = pooled_loader
            # temp_pooled = data_utils.TensorDataset(pooled_data, pooled_target)
            # args.pooled_loader = data_utils.DataLoader(temp_pooled, batch_size=args.batch_size, shuffle=True)

            args.layer_mix = None
            total_train_loss = 0.0
            args.policy_generator = PolicyGenerator(args).cuda()
            args.intrusion_discriminator = IntrusionDiscriminator(args).cuda()

            if args.isInit == 'True':
                init_model(args, first=False)
            print('...AdaMixup')
            for epoch in range(args.epochs2):
                # (1) Labeled data
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
                # (2) AdaMixup and Training
                train_loss_mixed, num_batch_mixed, num_data_mixed, image1, image2, image_mix = train_adamixed(args)
                total_train_loss += (train_loss_labeled+train_loss_mixed) / (num_data_labeled+num_data_mixed)

            total_train_loss /= args.epochs2

        if args.method == 'AcqOnly':
            pooled_data, pooled_target, pooled_target_oh = acquire_AcqOnly(args)

            args.train_data = torch.cat([args.train_data, pooled_data], 0)
            args.train_target = torch.cat([args.train_target, pooled_target], 0)
            if args.data == 'Cifar10' or args.data == 'Cifar100':
                transform_train = transforms.Compose(
                    [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
            else:
                transform_train = None
            train_dataset = CustomTensorDataset(tensors=(args.train_data, args.train_target), dataname=args.data, transform=transform_train)
            train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            args.train_target_oh = torch.cat([args.train_target_oh, pooled_target_oh], 0)
            args.train_loader = train_loader

            total_train_loss = 0.0
            if args.isInit == 'True':
                init_model(args, first=False)
            for epoch in range(args.epochs2):
                args.layer_mix = None
                train_loss_labeled, num_batch_labeled, num_data_labeled = train(epoch, args)
                total_train_loss += train_loss_labeled/num_data_labeled
            total_train_loss /= args.epochs2


        print('(step3) Results')
        args.layer_mix = None
        test_loss, test_acc, best_acc = test(i, args)
        args.test_acc_hist.append(test_acc)
        args.train_loss_hist.append(total_train_loss)
        et = time.time()

        print('...Train loss = %.5f' % total_train_loss)
        print('...Test accuracy = %.2f%%' % test_acc)
        print('...Best accuracy = %.2f%%' % args.best_acc)
        print_total_time(st, et)

    if not os.path.exists(args.saveDir):
        print('Result Directory Constructed Successfully!!!')
        os.makedirs(args.saveDir)

    np.save(args.saveDir + '/test_acc.npy', np.asarray(args.test_acc_hist))
    np.save(args.saveDir + '/train_loss.npy', np.asarray(args.train_loss_hist))

    return args.test_acc_hist


def max_entroy_acquisition(pool_data_dropout, pool_target_dropout, args):
    score_All = torch.tensor(torch.zeros(pool_data_dropout.size(0), args.nb_classes))

    data_size = pool_data_dropout.shape[0]
    num_batch = int(data_size / args.pool_batch_size)
    for idx in range(num_batch):
        batch = pool_data_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]
        target = pool_target_dropout[idx*args.pool_batch_size : (idx+1)*args.pool_batch_size]

        pool = data_utils.TensorDataset(batch, target)
        pool_loader = data_utils.DataLoader(pool, batch_size=args.pool_batch_size, shuffle=False)
        predictions = evaluate2('loader', pool_loader, None, args)
        predictions = predictions.cpu().detach()
        score_All[idx*args.pool_batch_size:(idx+1)*args.pool_batch_size, :] = predictions

    Avg_Pi = torch.div(score_All, args.di)
    Log_Avg_Pi = torch.log2(Avg_Pi)
    Entropy_Avg_Pi = - torch.mul(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = torch.sum(Entropy_Avg_Pi, 1)

    U_X = Entropy_Average_Pi

    points_of_interest = U_X.flatten()

    return points_of_interest

