import tensorflow as tf # stackoverflow.com/questions/73433868

import torch

import os
import glob
import argparse
from typing import Optional

import pytorch_lightning as pl

#from models.seq2seq import Seq2SeqModule
from models.seq2seq_ls import LSSeq2SeqModule as Seq2SeqModule
from models.vae import VqVaeModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--root_dir',
    type=str,
    default='./lmd_full',
    help='Dataset root dir.'
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    default='./results',
    help='The output dir of a given run.'
  )
  parser.add_argument(
    '--logging_dir',
    type=str,
    default='./logs',
    help='Directory for logging files.'
  )
  parser.add_argument(
    '--max_n_files',
    type=int,
    default=-1,
    help='Number if MIDI files to process.'
  )
  parser.add_argument(
    '--model',
    type=str,
    default="baseline",
    help='Name of the model.'
  )
  parser.add_argument(
    '--n_codes',
    type=int,
    default=2048,
    help='Number of codes.'
  )
  parser.add_argument(
    '--n_groups',
    type=int,
    default=16,
    help='Number of groups.'
  )
  parser.add_argument(
    '--d_model',
    type=int,
    default=512,
    help='Model dim.'
  )
  parser.add_argument(
    '--d_latent',
    type=int,
    default=1024,
    help='Number of latents dims.'
  )
  parser.add_argument(
    '--checkpoint',
    type=Optional[str],
    default=None,
    help='Path to a pre-trained seq2seq checkpoint.'
  )
  parser.add_argument(
    '--vae_checkpoint',
    type=Optional[str],
    default=None,
    help='Path to a pre-trained vq-vae checkpoint.'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Actual batch size on device.'
  )
  parser.add_argument(
    '--target_batch_size',
    type=int,
    default=512,
    help='Wanted batch size. Reached by accumulating batches.'
  )
  parser.add_argument(
    '--epochs',
    type=int,
    default=16,
    help='Number of training epochs.'
  )
  parser.add_argument(
    '--warmup_steps',
    type=int,
    default=4000,
    help='Number of warmup steps.'
  )
  parser.add_argument(
    '--max_steps',
    type=int,
    default=1e20,
    help='Maximum number of steps.'
  )
  parser.add_argument(
    '--max_training_steps',
    type=int,
    default=100_000,
    help='Maximum number of training steps.'
  )
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-4,
    help='Peak learning rate.'
  )
  parser.add_argument(
    '--lr_schedule',
    type=str,
    default='const',
    help='Type of the learning rate schedule.'
  )
  parser.add_argument(
    '--context_size',
    type=int,
    default=256,
    help='Size of sequence context.'
  )
  parser.add_argument(
    '--precision',
    type=int,
    default=16,
    help='Precision (16 or 32).'
  )
  parser.add_argument(
    '--amp_backend',
    type=str,
    default='apex',
    help='AMP backend for training with pytorch-lightning.'
  )
  parser.add_argument(
    '--amp_level',
    type=str,
    default='apex',
    help='AMP level for pl trainer.'
  )
  parser.add_argument(
    '--n_workers',
    type=int,
    default=8,
    help='Number of workers for the data pipeline.'
  )
  parser.add_argument(
    '--lightseq',
    action='store_true',
    default=False,
    help="Wether to use lightseq."
  )

  return parser.parse_args()


def main():
  ### Prepare all arguments ###
  args = arguments()

  ACCUMULATE_GRADS = max(1, args.target_batch_size//args.batch_size)
  N_WORKERS = min(os.cpu_count(), float(args.n_workers))
  if device.type == 'cuda':
    N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())
  N_WORKERS = int(N_WORKERS)


  ### Define available models ###
  available_models = [
    'vq-vae',
    'figaro-learned',
    'figaro-expert',
    'figaro',
    'figaro-inst',
    'figaro-chord',
    'figaro-meta',
    'figaro-no-inst',
    'figaro-no-chord',
    'figaro-no-meta',
    'baseline',
  ]

  assert args.model is not None, 'the MODEL needs to be specified'
  assert args.model in available_models, f'unknown MODEL: {args.model}'


  ### Create data loaders ###
  midi_files = glob.glob(os.path.join(args.root_dir, '**/*.mid'), recursive=True)
  if args.max_n_files > 0:
    midi_files = midi_files[:args.max_n_files]

  if len(midi_files) == 0:
    print(f"WARNING: No MIDI files were found at '{args.root_dir}'. Did you download the dataset to the right location?")
    exit()

  MAX_CONTEXT = min(1024, args.context_size)

  if args.model in ['figaro-learned', 'figaro'] and args.vae_checkpoint:
    vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=args.vae_checkpoint)
    vae_module.cpu()
    vae_module.freeze()
    vae_module.eval()

  else:
    vae_module = None


  ### Create and train model ###

  # load model from checkpoint if available

  if args.checkpoint:
    model_class = {
      'vq-vae': VqVaeModule,
      'figaro-learned': Seq2SeqModule,
      'figaro-expert': Seq2SeqModule,
      'figaro': Seq2SeqModule,
      'figaro-inst': Seq2SeqModule,
      'figaro-chord': Seq2SeqModule,
      'figaro-meta': Seq2SeqModule,
      'figaro-no-inst': Seq2SeqModule,
      'figaro-no-chord': Seq2SeqModule,
      'figaro-no-meta': Seq2SeqModule,
      'baseline': Seq2SeqModule,
    }[args.model]
    model = model_class.load_from_checkpoint(checkpoint_path=args.checkpoint)

  else:
    seq2seq_kwargs = {
      'encoder_layers': 4,
      'decoder_layers': 6,
      'num_attention_heads': 8,
      'intermediate_size': 2048,
      'd_model': args.d_model,
      'context_size': MAX_CONTEXT,
      'lr': args.learning_rate,
      'warmup_steps': args.warmup_steps,
      'max_steps': args.max_steps,
    }
    dec_kwargs = { **seq2seq_kwargs }
    dec_kwargs['encoder_layers'] = 0

    # use lambda functions for lazy initialization
    model = {
      'vq-vae': lambda: VqVaeModule(
        encoder_layers=4,
        decoder_layers=6,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        n_codes=args.n_codes, 
        n_groups=args.n_groups, 
        context_size=MAX_CONTEXT,
        lr=args.learning_rate,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        d_model=args.d_model,
        d_latent=args.d_latent,
      ),
      'figaro-learned': lambda: Seq2SeqModule(
        description_flavor='latent',
        n_codes=vae_module.n_codes,
        n_groups=vae_module.n_groups,
        d_latent=vae_module.d_latent,
        **seq2seq_kwargs
      ),
      'figaro': lambda: Seq2SeqModule(
        description_flavor='both',
        n_codes=vae_module.n_codes,
        n_groups=vae_module.n_groups,
        d_latent=vae_module.d_latent,
        **seq2seq_kwargs
      ),
      'figaro-expert': lambda: Seq2SeqModule(
        description_flavor='description',
        **seq2seq_kwargs
      ),
      'figaro-no-meta': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': True, 'chords': True, 'meta': False },
        **seq2seq_kwargs
      ),
      'figaro-no-inst': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': False, 'chords': True, 'meta': True },
        **seq2seq_kwargs
      ),
      'figaro-no-chord': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': True, 'chords': False, 'meta': True },
        **seq2seq_kwargs
      ),
      'baseline': lambda: Seq2SeqModule(
        description_flavor='none',
        **dec_kwargs
      ),
    }[args.model]()

  datamodule = model.get_datamodule(
    midi_files,
    vae_module=vae_module,
    batch_size=args.batch_size, 
    num_workers=N_WORKERS, 
    pin_memory=True
  )

  checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor='valid_loss',
    dirpath=os.path.join(args.output_dir, args.model),
    filename='{step}-{valid_loss:.2f}',
    save_last=True,
    save_top_k=2,
    every_n_train_steps=1000,
  )

  lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

  trainer = pl.Trainer(
    gpus=0 if device.type == 'cpu' else torch.cuda.device_count(),
    accelerator='gpu',
    profiler='simple',
    callbacks=[checkpoint_callback, lr_monitor],
    max_epochs=args.epochs,
    max_steps=args.max_training_steps,
    log_every_n_steps=max(100, min(25*ACCUMULATE_GRADS, 200)),
    val_check_interval=max(500, min(300*ACCUMULATE_GRADS, 1000)),
    precision=args.precision,
    limit_val_batches=64,
    auto_scale_batch_size=False,
    auto_lr_find=False,
    accumulate_grad_batches=ACCUMULATE_GRADS,
    stochastic_weight_avg=True,
    gradient_clip_val=1.0, 
    terminate_on_nan=True,
    resume_from_checkpoint=args.checkpoint
  )

  trainer.fit(model, datamodule)

if __name__ == '__main__':
  main()