import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from pointcloudutils.networks.vq_vae_pl import VQVAE
from hydra.utils import instantiate

import argparse
import math
import os.path
import wandb
import yaml



def set_matmul_precision():
    """
    If using Ampere Gpus enable using tensor cores
    """

    gpu_cores = os.popen('nvidia-smi -L').readlines()[0]

    if 'A100' in gpu_cores.lower():
        torch.set_float32_matmul_precision('high')
        print('[INFO] set matmul precision "high"')


def main():

    # only for A100
    set_matmul_precision()

    # configuration params (assumes some env variables in case of multi-node setup)
    num_nodes = int(os.getenv('NODES')) if os.getenv('NODES') is not None else 1
    gpus = torch.cuda.device_count()
    rank = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    is_dist = gpus > 1 or num_nodes > 1

    workers = 10
    seed = 42

    t_conf= {'cumulative_bs': 256, 'base_lr': 3e-4, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 1e-4, 'decay_epochs': 150, 
            'max_epochs': 150}

    cumulative_batch_size = int(t_conf['cumulative_bs'])
    batch_size_per_device = cumulative_batch_size // (num_nodes * gpus)

    base_learning_rate = float(t_conf['base_lr'])
    learning_rate = base_learning_rate * math.sqrt(cumulative_batch_size / 256)

    max_epochs = int(t_conf['max_epochs'])

    pl.seed_everything(seed, workers=True)

    # logging stuff, checkpointing and resume
    log_to_wandb = True
    project_name = 'test'
    wandb_id = 'run1'

    run_name = 'run_name1'
    save_checkpoint_dir = f'/allen/aics/modeling/ritvik/projects/npm1_checkpoints/'
    save_every_n_epochs = 15

    load_checkpoint_path = None
    resume = load_checkpoint_path is not None

    # if rank == 0:  # prevents from logging multiple times
    #     logger = WandbLogger(project=project_name, name=run_name, offline=not log_to_wandb, id=wandb_id,
    #                          resume='must' if resume else None)
    # else:
    #     logger = WandbLogger(project=project_name, name=run_name, offline=True)

    config = yaml.safe_load('''  
        _target_: cyto_dl.loggers.MLFlowLogger
        tracking_uri: https://mlflow.a100.int.allencell.org/
        experiment_name: 'pcna_image'
        run_name: 'vq_vae_2d_npm1'
    ''')

    logger = instantiate(config)


    # model params
    image_size = 32
    ae_conf = {'channels': 128, 'num_res_blocks': 2, 'channel_multipliers': [1,2,2,4]}
    q_conf = {'num_embeddings': 1024, 'embedding_dim': 256, 'type': 'standard', 'params': {'commitment_cost': 0.25}}
    l_conf = None
    t_conf = {'lr': 1e-3,
              'betas': t_conf['betas'],
              'eps': t_conf['eps'],
              'weight_decay': t_conf['weight_decay'],
              'warmup_epochs': t_conf['warmup_epochs'] if 'warmup_epochs' in t_conf.keys() else None,
              'decay_epochs': t_conf['decay_epochs'] if 'decay_epochs' in t_conf.keys() else None,
              }

    # check if using adversarial loss
    use_adversarial = l_conf is not None and 'adversarial_params' in l_conf.keys()

    # get model
    if resume:
        # image_size: int, ae_conf: dict, q_conf: dict, l_conf: dict, t_conf: dict, init_cb: bool = True,
        #                  load_loss: bool = True
        model = VQVAE.load_from_checkpoint(load_checkpoint_path, strict=False,
                                           image_size=image_size, ae_conf=ae_conf, q_conf=q_conf, l_conf=l_conf,
                                           t_conf=t_conf, init_cb=False, load_loss=True)
    else:
        model = VQVAE(image_size=image_size, ae_conf=ae_conf, q_conf=q_conf, l_conf=l_conf, t_conf=t_conf,
                      init_cb=True, load_loss=True)

    # data loading (standard pytorch lightning or ffcv)
    # datamodule = get_datamodule(args.dataloader, args.dataset_path, image_size, batch_size_per_device,
    #                             workers, seed, is_dist)

    with open("/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_npm1/variance_2d_npm1_align_maxproj.yaml", "r") as stream:
        config = yaml.safe_load(stream)    

    datamodule = instantiate(config)
    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=save_checkpoint_dir, filename='{epoch:02d}', save_last=True,
                                          save_top_k=-1, every_n_epochs=10)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=save_checkpoint_dir, filename='{epoch:02d}', save_last=True,
    #                                       save_top_k=-1, every_n_epochs=save_every_n_epochs)

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # trainer
    # set find unused parameters if using vqgan (adversarial training)
    trainer = pl.Trainer(accelerator='gpu', devices=[0],
                         callbacks=callbacks, deterministic=True, logger=logger, max_epochs=max_epochs)

    print(f"[INFO] workers: {workers}")
    print(f"[INFO] batch size per device: {batch_size_per_device}")
    print(f"[INFO] cumulative batch size (all devices): {cumulative_batch_size}")
    print(f"[INFO] final learning rate: {learning_rate}")

    # check to prevent later error
    if use_adversarial and batch_size_per_device % 4 != 0:
        raise RuntimeError('batch size per device must be divisible by 4! (due to stylegan discriminator forward pass)')

    # trainer.fit(model, datamodule, ckpt_path=load_checkpoint_path)
    trainer.fit(model, train_dataloaders=[datamodule.train_dataloader()],val_dataloaders=[datamodule.val_dataloader()], ckpt_path=load_checkpoint_path)

    # ensure wandb has stopped logging
    wandb.finish()


if __name__ == '__main__':
    main()