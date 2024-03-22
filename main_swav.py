# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
import matplotlib.pyplot as plt
from logging import getLogger
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torchvision.utils import save_image
# import apex
# from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger_ in loggers:
    logger_.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local-rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

# new buffering strategies
parser.add_argument("--buffer_strategy", type=str, default='fifo', help="options: \{fifo,element,prototype,code\}")
parser.add_argument("--no_grad", action='store_true', help="whether to use gradients")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        raise NotImplementedError
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    # if args.use_fp16:
    #     model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
    #     logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        #amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_x = None
    queue_age = None
    queue_target = None
    # queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    # if os.path.isfile(queue_path):
    #     queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()
            queue_x = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                3, args.size_crops[0], args.size_crops[0],
            ).cuda()
            queue_age = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
            ).cuda()
            queue_target = torch.zeros(
                args.queue_length // args.world_size,
                dtype=torch.long,
            ).cuda()
            

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, (queue, queue_x, queue_age, queue_target))
        training_stats.update(scores)

        # save checkpoints
        # if args.rank == 0:
        #     save_dict = {
        #         "epoch": epoch + 1,
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
            # if args.use_fp16:
            #     raise NotImplementedError
                # save_dict["amp"] = apex.amp.state_dict()
            # torch.save(
            #     save_dict,
            #     os.path.join(args.dump_path, "checkpoint.pth.tar"),
            # )
            # if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            #     shutil.copyfile(
            #         os.path.join(args.dump_path, "checkpoint.pth.tar"),
            #         os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
            #     )
        # if queue is not None:
        #     torch.save({"queue": queue}, queue_path)


def train(train_loader, model, optimizer, epoch, lr_schedule, queues):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    use_the_queue = False
    queue, queue_x, queue_age, queue_target = queues

    end = time.time()
    for it, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        with torch.amp.autocast(enabled=args.use_fp16, device_type='cuda'):
        
            # ============ multi-res forward passes ... ============
            with (torch.no_grad() if args.no_grad else torch.enable_grad()): 
                embedding, output = model(inputs)
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            if use_the_queue is False:
                                logger.info("Queue is now full, begin using queue.")
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.module.prototypes.weight.t()
                            ), out))
            
                    # get assignments
                    q = distributed_sinkhorn(out)        
                    
                    if queue is not None:
                        if use_the_queue == False or args.buffer_strategy == 'fifo':
                            queue, queue_x, queue_age, queue_target, w_evict_buffer = update_buffer((queue, queue_x, queue_age, queue_target),
                                                                                    inputs, target, embedding, bs, len(args.crops_for_assign), 'fifo')
                        else:
                            queue, queue_x, queue_age, queue_target, w_evict_buffer = update_buffer((queue, queue_x, queue_age, queue_target),
                                                                                    inputs, target, embedding, bs, len(args.crops_for_assign), args.buffer_strategy,
                                                                                    out, q, model.module.prototypes.weight)
                    
                    q = q[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                loss += subloss / (np.sum(args.nmb_crops) - 1)
            loss /= len(args.crops_for_assign)

        # ============ backward and optim step ... ============
        if not args.no_grad:
            optimizer.zero_grad()
            if args.use_fp16:
                raise NotImplementedError
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # cancel gradients for the prototypes
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 10 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            if not torch.all(queue[i, -1, :] == 0):
                logger.info(f"saving queue_{epoch}_{it}.png")
                save_image(queue_x.permute(1, 0, 2, 3, 4).flatten(0, 1)[:64], f"{args.dump_path}/queue_{epoch}_{it}.png", nrow=8)
            if w_evict_buffer is not None:
                plot_2d_heatmap(w_evict_buffer, f"{args.dump_path}/cosine_sim_{epoch}_{it}.png")
            plot_class_distribution(queue_target, f"{args.dump_path}/class_distribution_{epoch}_{it}.png")
            
    return (epoch, losses.avg), queue

def update_buffer(queues, x, y, z, N, num_crops_for_assign, mode, p=None, q=None, c=None):
    """update buffer

    Args:
        queues (tuple): queues that are to be updated
        x (list[torch.Tensor]): input images of shape [N, C, H, W]
        y (torch.Tensor): target labels of shape [N]
        z (torch.Tensor): encoder outputs of shape [N*nmb_crops, D]
        p (torch.Tensor): encoder outputs matmul'd with prototypes [N*nmb_crops+B, K]
        q (torch.Tensor): sinkhorn cluster assignments [N*nmb_crops+B, K]
        N (int): batch size
        num_crops_for_assign (int): number of crops for assign, usually 2
        mode (str): buffer update mode (fifo, element, prototype, code)
    """
    assert mode in ['fifo', 'element', 'prototype', 'code']
    
    queue, queue_x, queue_age, queue_target = queues
    D = z.shape[-1]
    B = queue.shape[1]
    device = z.device
    
    if mode == 'fifo':
        queue[:, N:] = queue[:, :-N].clone() # shift
        queue[:, :N] = z[:N*num_crops_for_assign].reshape(num_crops_for_assign, N, -1) # update
        queue_x[:, N:] = queue_x[:, :-N].clone() # shift
        queue_x[:, :N] = torch.stack(x[:num_crops_for_assign], dim=0).cuda(non_blocking=True).reshape(num_crops_for_assign, N, *x[0].shape[-3:]) # update
        queue_age += 1 # increment
        queue_age[:, N:] = queue_age[:, :-N].clone() # shift
        queue_age[:, :N] = 0 # reset
        queue_target[N:] = queue_target[:-N].clone() # shift
        queue_target[:N] = y # update
        return queue, queue_x, queue_age, queue_target, None
    else:
        bz = torch.cat((queue, z[:N*num_crops_for_assign].reshape(num_crops_for_assign, N, -1)), dim=1) # nmb_crops_for_assign, N + B, D
        x_ = torch.stack(x[:num_crops_for_assign], dim=0).cuda(non_blocking=True).reshape(num_crops_for_assign, N, *x[0].shape[-3:])
        qxx = torch.cat((queue_x, x_), dim=1) # nmb_crops_for_assign, N + B, 3, H, W
        age_ = torch.cat((queue_age + 1., torch.zeros(num_crops_for_assign, N, dtype=torch.long, device=device)), dim=1) # nmb_crops_for_assign, N + B
        qyy = torch.cat((queue_target, y.cuda(non_blocking=True)), dim=0) # N + B
        
        if mode == 'element':
            cosine_sim = torch.bmm(bz, bz.permute(0, 2, 1)) + 1. - torch.eye(N+B, device=device) * 2 # nmb_crops_for_assign, N + B, N + B
            w_evict_bz = cosine_sim.sum(dim=2).mean(0) # N + B
            evictions = torch.multinomial(w_evict_bz + 1e-7, N, replacement=False)
        if mode == 'prototype':
            K = c.shape[0]
            c_norm = F.normalize(c, p=2, dim=1) # K, D
            cluster_sim = torch.mm(c_norm, c_norm.t()) + 1. - torch.eye(K, device=device) * 2 # K, K
            w_evict_cluster = cluster_sim.sum(dim=1) / cluster_sim.sum() # K
            w_evict_bz = ((p + 1.) * w_evict_cluster[None]).sum(dim=1) # N + B
            evictions = torch.multinomial(w_evict_bz + 1e-7, N, replacement=False)
            
        unevicted_indices = torch.ones(N + queue.shape[1], dtype=torch.bool, device=device)
        unevicted_indices[evictions] = False
        
        queue[:, :] = bz[:, unevicted_indices] # update
        queue_x[:, :] = qxx[:, unevicted_indices] # update
        queue_age[:, :] = age_[:, unevicted_indices] # update
        queue_target[:] = qyy[unevicted_indices] # update  
        return queue, queue_x, queue_age, queue_target, w_evict_bz

def plot_2d_heatmap(p, f):
    N = p.shape[0]
    probabilities = p.cpu().numpy()

    # Reshape probabilities to 2D array
    probabilities = probabilities.reshape(8, -1)

    # Plotting heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(probabilities, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Probability Heatmap')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.tight_layout()

    # Save plot to file
    plt.savefig(f)

def plot_class_distribution(class_labels_tensor, filepath=None):
    # Convert tensor to numpy array for plotting
    class_labels_np = class_labels_tensor.cpu().numpy()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(class_labels_np, bins=range(int(class_labels_tensor.max()) + 2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Class Labels')
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.xticks(range(int(class_labels_tensor.max()) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if filepath:
        plt.savefig(filepath)  # Save the plot to the specified file
    else:
        plt.show()

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()
