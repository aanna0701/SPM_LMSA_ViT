# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(args, optimizer, n_iter_per_epoch):
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        t_mul=1.,
        lr_min=5e-6,
        warmup_lr_init=5e-7,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        )
    return lr_scheduler
