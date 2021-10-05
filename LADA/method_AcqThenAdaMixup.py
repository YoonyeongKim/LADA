import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import truncnorm
import numpy as np

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return torch.mean(lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b))

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def train_adamixed(args):
    args.model.train()
    args.policy_generator.train()
    args.intrusion_discriminator.train()
    alpha_list = []
    delta_list = []
    intrusion_loss_list = []
    class_loss_list = []
    num_data = 0
    num_batch = 0
    for batch_idx, (data, target) in enumerate(args.pooled_loader):
        ''' preapare the basic dataset '''
        data, target = data.cuda(), target.cuda()
        batch, target = Variable(data), Variable(target)

        batch_size = data.shape[0]
        index = torch.randperm(batch_size).cuda()

        image_1 = data
        image_2 = data[index]
        label_1 = target
        label_2 = target[index]

        image_1_f, image_1_s = torch.chunk(image_1, chunks=2, dim=0)
        image_2_f, image_2_s = torch.chunk(image_2, chunks=2, dim=0)
        label_1_f, label_1_s = torch.chunk(label_1, chunks=2, dim=0)
        label_2_f, label_2_s = torch.chunk(label_2, chunks=2, dim=0)

        image_1 = image_1_f
        image_2 = image_2_f
        label_1 = label_1_f
        label_2 = label_2_f

        image_unseen = image_1_s
        label_unseen = label_1_s

        args.optimizer.zero_grad()

        # 1) make gamma (policy region generator)
        m1 = torch.from_numpy(truncated_normal(image_1.shape, threshold=2)).float().cuda()
        m2 = torch.from_numpy(truncated_normal(image_1.shape, threshold=2)).float().cuda()
        image_gate = (m1 * image_1 + m2 * image_2) / 2.0
        policy_output = args.policy_generator(image_gate)
        alpha, delta, alpha_prime = torch.chunk(policy_output, 3, dim=1)
        alpha_list.append(alpha.detach().cpu().numpy())
        delta_list.append(delta.detach().cpu().numpy())
        eps = args.sampler.sample(sample_shape=alpha.shape).cuda()
        eps = torch.squeeze(eps, -1)
        gamma = delta * eps + alpha
        gamma = torch.unsqueeze(gamma, -1)
        gamma = torch.unsqueeze(gamma, -1)

        # 2) image classification
        image_mix = torch.mul(image_1, gamma) + torch.mul(image_2, (1 - gamma))
        new_inputs = torch.cat([image_1, image_mix], dim=0)
        new_y1 = torch.cat([label_1, label_1], dim=0)
        new_y2 = torch.cat([label_2, label_2], dim=0)
        new_gamma = torch.cat([torch.ones_like(gamma), gamma], dim=0).squeeze(-1)
        new_gamma = torch.squeeze(new_gamma, -1)
        new_gamma = torch.squeeze(new_gamma, -1)

        bef, class_output = args.model(new_inputs)
        loss1 = mixup_criterion(args.criterion, class_output, new_y1, new_y2, new_gamma)

        # 3) intrusion
        binary_images_pos = torch.cat([image_unseen, image_1], dim=0)
        binary_images_neg = torch.cat([image_mix], dim=0)
        labels_pos = torch.cat([torch.ones_like(label_1), torch.ones_like(label_1)], dim=0)
        labels_neg = torch.zeros_like(label_1)

        binary_images = torch.cat([binary_images_pos, binary_images_neg], dim=0)
        binary_labels = torch.cat([labels_pos, labels_neg], dim=0)

        bef, output = args.model(binary_images)
        intrusion_output = args.intrusion_discriminator(bef)
        loss2 = args.intrusion_criterion(intrusion_output, binary_labels)

        class_loss_list.append(loss1.item())
        intrusion_loss_list.append(loss2.item())
        # 4) iterative optimization
        loss = loss1 + loss2
        loss1.backward(retain_graph=True)
        args.optimizer.step()
        args.optimizer.zero_grad()
        loss2.backward()
        args.optimizer.step()

        num_data += image_mix.shape[0]
        num_batch += 1

    return loss1.item() + loss2.item(), num_data, num_batch, image_1, image_2, image_mix