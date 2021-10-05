from utils_method import *
import numpy as np
import random
from math import ceil

def train_lambdanet_vanilla_featuremap(pool_data_dropout, pool_target_dropout, shuffled_pool_data, shuffled_pool_target, generator, args):

    data_size = pool_data_dropout.shape[0]
    num_batch = ceil(data_size / args.pool_batch_size)

    for itr in range(args.numLambda):
        for idx in range(num_batch):
            batch1 = pool_data_dropout[idx * args.pool_batch_size: (idx+1) * args.pool_batch_size]
            target1 = pool_target_dropout[idx * args.pool_batch_size: (idx+1) * args.pool_batch_size]
            batch2 = shuffled_pool_data[idx * args.pool_batch_size: (idx+1) * args.pool_batch_size]
            target2 = shuffled_pool_target[idx * args.pool_batch_size: (idx+1) * args.pool_batch_size]
            mixed_batch, mixed_target, a = mixup_two_batches(batch1, target1, batch2, target2, args)
            mixed_predictions = evaluate_featuremap(mixed_batch, args)
            mixed_log_predictions = torch.log2(mixed_predictions)
            Entropy_Avg_Pi = -torch.mul(mixed_predictions, mixed_log_predictions)  # batch_size x nb_classes
            Entropy_Average_Pi = torch.sum(Entropy_Avg_Pi, 1)  # batch_size
            temp_entropy_final = Entropy_Average_Pi.unsqueeze_(1).cpu().detach()  # pool_data_size x 1
            if idx == 0:
                entropy_final = temp_entropy_final  # pool_data_size x 1
            else:
                entropy_final = torch.cat([entropy_final, temp_entropy_final], 0)
        if itr == 0:
            lsEntropy = entropy_final
        else:
            lsEntropy = torch.cat([lsEntropy, entropy_final], 1)

    avg_lsEntropy = torch.mean(lsEntropy, 1)

    return avg_lsEntropy


def acquire_ManifoldMixup_vanilla(pool_data_dropout_featuremap, pool_target_dropout, shuffled_pool_data_featuremap, shuffled_pool_target, args):
    # Train lambdanet and generate new data
    args.input_dim_featuremap = pool_data_dropout_featuremap.shape[1]
    args.input_height_featuremap = pool_data_dropout_featuremap.shape[2]
    args.input_width_featuremap = pool_data_dropout_featuremap.shape[3]

    generator = lambdanet(args).cuda()
    points_of_interest_mixup = train_lambdanet_vanilla_featuremap(pool_data_dropout_featuremap, pool_target_dropout, shuffled_pool_data_featuremap, shuffled_pool_target, generator, args)

    return points_of_interest_mixup

def acquire_VanillaManifoldMixupAcq(args):
    args.model.eval()
    # 1-1) Sampling
    pool_subset_dropout = torch.from_numpy(
        np.asarray(random.sample(range(0, args.pool_data.size(0)), args.pool_subset))).long()

    pool_data_dropout = args.pool_data[pool_subset_dropout]
    pool_target_dropout = args.pool_target[pool_subset_dropout]

    reshaped_pool_subset_dropout = torch.reshape(pool_subset_dropout, (2, int(args.pool_subset / 2)))
    pool_subset_dropout1 = reshaped_pool_subset_dropout[0]
    pool_subset_dropout2 = reshaped_pool_subset_dropout[1]

    pool_data_dropout1 = args.pool_data[pool_subset_dropout1]
    pool_target_dropout1 = args.pool_target[pool_subset_dropout1]

    pool_data_dropout2 = args.pool_data[pool_subset_dropout2]
    pool_target_dropout2 = args.pool_target[pool_subset_dropout2]

    # 1-2) Acquisition
    points_of_interest1 = args.acquisition_function(pool_data_dropout1, pool_target_dropout1, args)
    points_of_interest2 = args.acquisition_function(pool_data_dropout2, pool_target_dropout2, args)
    args.layer_mix = random.randint(args.rangeLayerMixFrom, args.rangeLayerMixTo)
    layer_mix = args.layer_mix
    print('...VanillaManifoldMixup at layer %d'%args.layer_mix)
    pool_data_dropout1_featuremap = get_kth_feature_map(pool_data_dropout1, args)
    pool_data_dropout2_featuremap = get_kth_feature_map(pool_data_dropout2, args)

    pool_data_dropout1_featuremap = pool_data_dropout1_featuremap.detach().cpu()
    pool_data_dropout2_featuremap = pool_data_dropout2_featuremap.detach().cpu()
    points_of_interest_mixup = acquire_ManifoldMixup_vanilla(pool_data_dropout1_featuremap, pool_target_dropout1,
                                                              pool_data_dropout2_featuremap, pool_target_dropout2, args)

    points_of_interest1 = points_of_interest1.detach().cpu().numpy()
    points_of_interest2 = points_of_interest2.detach().cpu().numpy()
    points_of_interest_mixup = points_of_interest_mixup.detach().cpu().numpy()
    joint_points_of_interest = points_of_interest1 + points_of_interest2 + points_of_interest_mixup

    temp_pool_index1 = np.flip(joint_points_of_interest.argsort()[::-1][:int(args.numQ / 2)], axis=0)
    temp_pool_index2 = temp_pool_index1 + int(args.pool_subset / 2)
    pool_index = np.concatenate((temp_pool_index1, temp_pool_index2), 0)

    temp_pool_index1 = torch.from_numpy(temp_pool_index1)
    temp_pool_index2 = torch.from_numpy(temp_pool_index2).long()
    pool_index = torch.from_numpy(pool_index)

    pooled_data1 = pool_data_dropout[temp_pool_index1]
    pooled_target1 = pool_target_dropout[temp_pool_index1]
    pooled_data2 = pool_data_dropout[temp_pool_index2]
    pooled_target2 = pool_target_dropout[temp_pool_index2]

    pooled_data = torch.cat((pooled_data1, pooled_data2), 0)
    pooled_target = torch.cat((pooled_target1, pooled_target2), 0)

    batch_size = pooled_data.shape[0]
    target1 = pooled_target.unsqueeze(1)
    y_onehot1 = torch.FloatTensor(batch_size, args.nb_classes)
    y_onehot1.zero_()
    target1_oh = y_onehot1.scatter_(1, target1, 1)
    pooled_target_oh = target1_oh.float()

    # 1-3) Remove from pool_data
    pool_data, pool_target = remove_pooled_points(args.pool_data, args.pool_target, pool_subset_dropout,
                                                  pool_data_dropout, pool_target_dropout, pool_index)
    args.pool_data = pool_data
    args.pool_target = pool_target
    args.pool_all = np.append(args.pool_all, pool_index)

    return pooled_data, pooled_target, pooled_target_oh, pooled_data1, pooled_target1, pooled_data2, pooled_target2, layer_mix