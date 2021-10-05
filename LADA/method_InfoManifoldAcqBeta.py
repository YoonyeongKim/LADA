from math import ceil
import torch.optim as optim
import torch.distributions as distr
from utils_method import *
import numpy as np
import random

def mixup_two_batches_given_lambda_distr_Beta(batch1, target1, batch2, target2, out, args):

    out_cpu = out.cpu()
    f = distr.beta.Beta(out_cpu, out_cpu)
    lam_cpu = f.rsample()
    lam = lam_cpu.cuda()

    mixed_batch, mixed_target = mixup_two_batches_given_lambdas(batch1, target1, batch2, target2, lam, args)

    return mixed_batch, mixed_target


def train_lambdanet_info_featuremap_Beta(pool_data_dropout, pool_target_dropout, shuffled_pool_data, shuffled_pool_target, generator, args):

    def get_GLoss_Beta(batch1, target1, batch2, target2, out):

        for itr in range(args.numLambda):
            mixed_batch, _ = mixup_two_batches_given_lambda_distr_Beta(batch1, target1, batch2, target2, out, args)
            mixed_predictions = evaluate_featuremap(mixed_batch, args)
            mixed_predictions += 1e-100
            mixed_log_predictions = torch.log2(mixed_predictions)
            Entropy_Avg_Pi = -torch.mul(mixed_predictions, mixed_log_predictions)
            Entropy_Average_Pi = torch.sum(Entropy_Avg_Pi, 1)
            entropy_final = Entropy_Average_Pi.unsqueeze_(1)
            if itr == 0:
                lsEntropy = entropy_final
            else:
                lsEntropy = torch.cat([lsEntropy, entropy_final], 1)
        avg_lsEntropy = torch.mean(lsEntropy, 1)
        G_loss = -torch.sum(avg_lsEntropy)
        return G_loss

    entropies = []
    outs = []
    G_losses = []
    data_size = pool_data_dropout.shape[0]
    num_batch = ceil(data_size / args.pool_batch_size)
    generator = lambdanet(args).cuda()
    generator.train()
    G_optimizer = optim.Adam(generator.parameters(), lr=5e-5)
    min_Gloss = 999999

    for epoch in range(args.Gepochs):
        total_Gloss = 0.0
        idx_shuffle = torch.randperm(data_size)
        inverse_idx_shuffle = torch.argsort(idx_shuffle)
        pool_data_dropout2 = pool_data_dropout[idx_shuffle]
        pool_target_dropout2 = pool_target_dropout[idx_shuffle]
        shuffled_pool_data2 = shuffled_pool_data[idx_shuffle]
        shuffled_pool_target2 = shuffled_pool_target[idx_shuffle]
        for idx in range(num_batch):
            batch1 = pool_data_dropout2[idx * args.pool_batch_size: (idx + 1) * args.pool_batch_size]
            target1 = pool_target_dropout2[idx * args.pool_batch_size: (idx + 1) * args.pool_batch_size]
            batch2 = shuffled_pool_data2[idx * args.pool_batch_size: (idx + 1) * args.pool_batch_size]
            target2 = shuffled_pool_target2[idx * args.pool_batch_size: (idx + 1) * args.pool_batch_size]

            cat_batch = torch.cat((batch1, batch2), 1).cuda()
            batch1, batch2, target1, target2 = batch1.cuda(), batch2.cuda(), target1.cuda(), target2.cuda()

            G_optimizer.zero_grad()
            out_temp = generator(cat_batch, args)
            out = out_temp * args.sharp
            G_loss = get_GLoss_Beta(batch1, target1, batch2, target2, out)
            G_loss.backward()
            G_optimizer.step()
            total_Gloss += G_loss.item()

            if idx == 0:
                total_out = out
            else:
                total_out = torch.cat((total_out, out), 0)
        total_Gloss /= num_batch
        G_losses.append(total_Gloss)

        if total_Gloss < min_Gloss:
            min_Gloss = total_Gloss
            out_final = total_out[inverse_idx_shuffle]

    for idx in range(num_batch):
        batch1 = pool_data_dropout[idx * args.pool_batch_size : (idx + 1) * args.pool_batch_size]
        target1 = pool_target_dropout[idx * args.pool_batch_size : (idx + 1) * args.pool_batch_size]
        batch2 = shuffled_pool_data[idx * args.pool_batch_size : (idx + 1) * args.pool_batch_size]
        target2 = shuffled_pool_target[idx * args.pool_batch_size : (idx + 1) * args.pool_batch_size]
        batch1, batch2, target1, target2 = batch1.cuda(), batch2.cuda(), target1.cuda(), target2.cuda()
        temp_out_final = out_final[idx * args.pool_batch_size : (idx + 1) * args.pool_batch_size]

        for itr in range(args.numLambda):
            mixed_batch, _ = mixup_two_batches_given_lambda_distr_Beta(batch1, target1, batch2, target2, temp_out_final,
                                                                       args)
            mixed_predictions = evaluate_featuremap(mixed_batch, args)
            mixed_log_predictions = torch.log2(mixed_predictions)
            Entropy_Avg_Pi = -torch.mul(mixed_predictions, mixed_log_predictions)  # batch_size x nb_classes
            Entropy_Average_Pi = torch.sum(Entropy_Avg_Pi, 1)  # batch_size
            entropy_final = Entropy_Average_Pi.unsqueeze_(1).cpu().detach()  # pool_data_size x 1
            if itr == 0:
                lsEntropy = entropy_final
            else:
                lsEntropy = torch.cat([lsEntropy, entropy_final], 1)
        avg_lsEntropy = torch.mean(lsEntropy, 1)

        entropies.append(avg_lsEntropy.cpu().detach().numpy())
        outs.append(temp_out_final.cpu().detach().numpy())

    flat_entropies = [item for sublist in entropies for item in sublist]  # pool_data_size x 1
    flat_outs = [item for sublist in outs for item in sublist]

    return flat_entropies, flat_outs, G_losses

def acquire_ManifoldMixup_info_Beta(pool_data_dropout_featuremap, pool_target_dropout, shuffled_pool_data_featuremap, shuffled_pool_target, args):
    # Train lambdanet and generate new data
    args.input_dim_featuremap = pool_data_dropout_featuremap.shape[1]
    args.input_height_featuremap = pool_data_dropout_featuremap.shape[2]
    args.input_width_featuremap = pool_data_dropout_featuremap.shape[3]

    generator = lambdanet(args).cuda()
    points_of_interest_mixup, outs, G_losses = train_lambdanet_info_featuremap_Beta(pool_data_dropout_featuremap, pool_target_dropout, shuffled_pool_data_featuremap, shuffled_pool_target, generator, args)

    return points_of_interest_mixup, outs, G_losses

def acquire_InfoManifoldMixupAcq_Beta(args):
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
    print('...InfoManifoldMixup_Beta at layer %d'%args.layer_mix)
    pool_data_dropout1_featuremap = get_kth_feature_map(pool_data_dropout1, args)
    pool_data_dropout2_featuremap = get_kth_feature_map(pool_data_dropout2, args)

    pool_data_dropout1_featuremap = pool_data_dropout1_featuremap.detach().cpu()
    pool_data_dropout2_featuremap = pool_data_dropout2_featuremap.detach().cpu()
    points_of_interest_mixup, outs, G_losses = acquire_ManifoldMixup_info_Beta(pool_data_dropout1_featuremap, pool_target_dropout1,
                                                              pool_data_dropout2_featuremap, pool_target_dropout2, args)


    points_of_interest1 = points_of_interest1.detach().cpu().numpy()
    points_of_interest2 = points_of_interest2.detach().cpu().numpy()
    joint_points_of_interest = points_of_interest1 + points_of_interest2 + points_of_interest_mixup

    temp_pool_index1 = np.flip(joint_points_of_interest.argsort()[::-1][:int(args.numQ / 2)], axis=0)
    temp_pool_index2 = temp_pool_index1 + int(args.pool_subset / 2)
    pool_index = np.concatenate((temp_pool_index1, temp_pool_index2), 0)

    temp_pool_index1 = torch.from_numpy(temp_pool_index1).long()
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

    return pooled_data, pooled_target, pooled_target_oh, pooled_data1, pooled_target1, pooled_data2, pooled_target2, outs, temp_pool_index1, layer_mix
