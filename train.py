from data.dataset import ContentDatasetFromObj, StyleDatasetFromObj
from torch.utils.data import DataLoader
import torch.nn as nn
from model.model import EMD
import os
import torch
import random
import time
import math
import argparse


def chkormakedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def update_lr(optimizer_params):
    for p in optimizer_params:
        p['lr'] = max(p['lr'] / 2.0, 0.0002)


def main(
    experiment_dir='experiment',
    content_data_dir=None,
    style_data_dir=None,
    check_point_dir=None,
    sample_dir=None,
    img_size=256,
    content_input_nc=5,
    style_input_nc=5,
    ngf=64,
    batch_size=32,
    epoch=100,
    start_epoch=0,
    resume=None,
    lr=0.001,
    schedule=10,
    sample_steps=20,
    check_point_steps=20,
    random_seed=123,
    gpu_id='cuda',
):

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if content_data_dir is None:
        content_data_dir = os.path.join(experiment_dir, "content_data")
    if style_data_dir is None:
        style_data_dir = os.path.join(experiment_dir, "style_data")
    if check_point_dir is None:
        check_point_dir = os.path.join(experiment_dir, "check_point")
        chkormakedir(check_point_dir)
    if sample_dir is None:
        sample_dir = os.path.join(experiment_dir, "sample")
        chkormakedir(sample_dir)

    start_time = time.time()

    model = EMD(
        content_input_nc=content_input_nc,
        style_input_nc=style_input_nc,
        ngf=ngf,
        gpu_id=gpu_id,
        is_training=True)
    if gpu_id:
        model.to(gpu_id)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    if gpu_id:
        l1_loss.cuda()

    model.print_networks(True)

    if resume:
        model.load_networks(resume)

    # val dataset load only once, no shuffle
    val_content_dataset = ContentDatasetFromObj(
        os.path.join(
            content_data_dir,
            'val.obj'))
    val_style_dataset = StyleDatasetFromObj(
        os.path.join(
            style_data_dir,
            'val.obj'))
    val_content_dataloader = DataLoader(
        val_content_dataset,
        batch_size=batch_size,
        shuffle=False)
    val_style_dataloader = DataLoader(
        val_style_dataset,
        batch_size=batch_size,
        shuffle=False)

    for epoch in range(epoch):
        epoch += start_epoch
        # generate train dataset every epoch so that different styles of saved
        # char imgs can be trained.

        train_content_dataset = ContentDatasetFromObj(
            os.path.join(content_data_dir, 'train.obj'))
        train_style_dataset = StyleDatasetFromObj(
            os.path.join(style_data_dir, 'train.obj'))

        total_batches = math.ceil(len(train_content_dataset) / batch_size)
        content_dataloader = DataLoader(
            train_content_dataset,
            batch_size=batch_size,
            shuffle=True)

        style_dataloader_iter = DataLoader(
            train_style_dataset,
            batch_size=batch_size,
            shuffle=True).__iter__()

        for bid, content_batch in enumerate(content_dataloader):
            style_batch = next(style_dataloader_iter).to(gpu_id)
            if len(style_batch) < batch_size:
                style_dataloader_iter = DataLoader(
                    train_style_dataset,
                    batch_size=batch_size,
                    shuffle=True).__iter__()
                style_batch = next(style_dataloader_iter)
            ground_truth = content_batch[0].to(gpu_id)
            x = model(content_batch[1].to(gpu_id), style_batch)
            loss = l1_loss(ground_truth, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if bid % 500 == 0:
                passed = time.time() - start_time
                print(
                    f'Epoch: {epoch}, bid: {bid}, total_batches: {total_batches}, passed: {passed}, L1 Loss: {loss.item()}')

            if epoch % check_point_steps == 0:
                model.save_networks(
                    save_dir=check_point_dir,
                    epoch=epoch)
                print("Checkpoint: save checkpoint step %d" % epoch)

            if epoch % sample_steps == 0:
                val_style_dataloader_iter = val_style_dataloader.__iter__()
                for vbid, val_batch in enumerate(val_content_dataloader):
                    style_val = next(val_style_dataloader_iter).to(gpu_id)
                    if len(style_val) < batch_size:
                        val_style_dataloader_iter = val_style_dataloader.__iter__()
                        style_val = next(val_style_dataloader_iter)
                    ground_truth_val = val_batch[0]
                    content_val = val_batch[1].to(gpu_id)
                    basename = os.path.join(sample_dir, str(epoch))
                    chkormakedir(basename)
                    model.sample(
                        content_val, style_val, basename)
                print("Sample: sample step %d" % epoch)
        if (epoch + 1) % schedule == 0:
            update_lr(optimizer_params=optimizer.param_groups)

    val_style_dataloader_iter = val_style_dataloader.__iter__()
    for vbid, val_batch in enumerate(val_content_dataloader):
        style_val = next(val_style_dataloader_iter).to(gpu_id)
        ground_truth_val = val_batch[0]
        content_val = val_batch[1].to(gpu_id)
        basename = os.path.join(sample_dir, str(epoch))
        chkormakedir(basename)
        model.sample(
            content_val, style_val, basename)
        print("Checkpoint: save checkpoint step %d" % epoch)

    model.save_networks(save_dir=check_point_dir, epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
        '--experiment_dir',
        default='experiment',
        help='experiment directory, content_data, style_data samples,checkpoints,etc')
    parser.add_argument(
        '--content_data_dir',
        default='content_data',
        help='experiment directory, content_data, samples,checkpoints,etc')
    parser.add_argument(
        '--style_data_dir',
        default='style_data',
        help='experiment directory, content_data, samples,checkpoints,etc')
    parser.add_argument(
        '--check_point_dir',
        default='check_point',
        help='experiment directory, content_data, samples,checkpoints,etc')
    parser.add_argument('--gpu_id', default='cuda', help="GPUs")
    parser.add_argument('--image_size', type=int, default=256,
                        help="size of your input and output image")
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='number of epoch')
    parser.add_argument(
        '--start_epoch',
        type=int,
        default=100,
        help='number of start epoch')
    parser.add_argument(
        '--ngf',
        type=int,
        default=64,
        help='number of channels of the first layer')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='number of examples in batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate for adam')
    parser.add_argument(
        '--schedule',
        type=int,
        default=10,
        help='number of epochs to half learning rate')
    parser.add_argument(
        '--sample_steps',
        type=int,
        default=10,
        help='number of batches in between two samples are drawn from validation set')
    parser.add_argument('--check_point_steps', type=int, default=20,
                        help='number of batches in between two checkpoints')
    parser.add_argument('--random_seed', type=int, default=123,
                        help='random seed for random and pytorch')
    parser.add_argument('--resume', type=int, default=None,
                        help='resume from previous training')
    parser.add_argument('--content_input_nc', type=int, default=5,
                        help='number of input images channels')
    parser.add_argument('--style_input_nc', type=int, default=5,
                        help='number of input images channels')
    args = parser.parse_args()
    main(
        experiment_dir=args.experiment_dir,
        content_data_dir=args.content_data_dir,
        style_data_dir=args.style_data_dir,
        check_point_dir=args.check_point_dir,
        sample_dir=args.sample_dir,
        img_size=args.image_size,
        content_input_nc=args.content_input_nc,
        style_input_nc=args.style_input_nc,
        ngf=args.ngf,
        batch_size=args.batch_size,
        epoch=args.epoch,
        start_epoch=args.start_epoch,
        resume=args.resume,
        lr=args.lr,
        schedule=args.schedule,
        sample_steps=args.sample_steps,
        check_point_steps=args.check_point_steps,
        random_seed=args.random_seed,
        gpu_id=args.gpu_id,
    )
