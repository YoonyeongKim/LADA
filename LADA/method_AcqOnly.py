import random
from utils_method import *

def acquire_AcqOnly(args):
    args.model.eval()

    # 1-1) Sampling
    print('...Acquisition Only')
    pool_subset_dropout = torch.from_numpy(
        np.asarray(random.sample(range(0, args.pool_data.size(0)), args.pool_subset))).long()
    pool_data_dropout = args.pool_data[pool_subset_dropout]
    pool_target_dropout = args.pool_target[pool_subset_dropout]

    # 1-2) Acquisition
    points_of_interest = args.acquisition_function(pool_data_dropout, pool_target_dropout, args)
    points_of_interest = points_of_interest.detach().cpu().numpy()
    pool_index = np.flip(points_of_interest.argsort()[::-1][:int(args.numQ)], axis=0)

    pool_index = torch.from_numpy(pool_index)
    pooled_data = pool_data_dropout[pool_index]
    pooled_target = pool_target_dropout[pool_index]

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

    return pooled_data, pooled_target, pooled_target_oh