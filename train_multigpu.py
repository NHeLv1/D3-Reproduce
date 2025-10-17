import random
import argparse
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import util
from util import build_model, train_one_epoch_ddp
from dataloader import generate_dataset_loader_ddp
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of detector in yaml format')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    return args


def main_worker(rank, world_size, cfg):
    """Main worker function for each GPU process."""
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print("******* Building models. *******")
    model = util.build_model(cfg['model'])
    model = model.to(device)

    if cfg['tuning_mode'] == 'lp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
    loss = nn.BCEWithLogitsLoss()
    
    trMaxEpoch = cfg['max_epoch']
    snapshot_path = cfg['save_dir']
    
    # Only create directories on rank 0
    if rank == 0:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    max_epoch, max_acc = 0, 0

    for epochID in range(0, trMaxEpoch):
        if rank == 0:
            print("******* Training epoch", str(epochID)," *******")
            print("******* Building datasets. *******")
        
        train_loader, val_loader = generate_dataset_loader_ddp(cfg, world_size, rank)
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epochID)
        
        max_epoch, max_acc, epoch_time = train_one_epoch_ddp(
            cfg, model, loss, scheduler, optimizer, epochID, max_epoch, max_acc, 
            train_loader, val_loader, snapshot_path, rank, world_size
        )
        
        if rank == 0:
            print("******* Ending epoch", str(epochID)," Time ", str(epoch_time), "*******")

    cleanup()


def main():
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    world_size = torch.cuda.device_count()
    if args.world_size > 0:
        world_size = min(args.world_size, world_size)
    
    print(f"Using {world_size} GPUs for training")
    print(cfg)
    
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        # Single GPU fallback
        main_worker(0, 1, cfg)


if __name__ == '__main__': 
    main()