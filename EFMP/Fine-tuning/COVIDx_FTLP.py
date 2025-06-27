
# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import models_vit
from timm.models.layers import trunc_normal_

logger = logging.getLogger(__name__)

COVIDX_CLASS_NAMES = ['Normal', 'COVID-19']  # Assumed, please confirm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def auc(pred_property_array, one_hot_labels, num_classes):
    AUROCs = []
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(one_hot_labels[:, i], pred_property_array[:, i]))
    return AUROCs

def simple_accuracy(preds, labels):
    return ((preds == labels) * 1).mean()

def save_model(args, model, metric, mode):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_bestacc_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Saved model checkpoint to [DIR: {args.output_dir}] with accuracy: {metric:.5f}")

def load_weights(model, weight_path, args):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    if args.stage == 'train':
        pretrained_weights = pretrained_weights['model']
    model_weights = model.state_dict()
    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model

def setup(args):
    args.is_multilabel = False  # COVIDx is single-label binary classification
    num_classes = args.num_classes
    
    model = models_vit.__dict__[args.model](
        num_classes=num_classes,
        drop_path_rate=0.1,
        global_pool=True,
    )
    if args.stage == 'train':
        checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)
    else:
        args.pretrained_path = os.path.join(args.output_dir, f"{args.name}_bestacc_checkpoint.bin")
        model = load_weights(model, args.pretrained_path, args)

    if args.mode == "LinearProbe":
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    model.to(args.device)
    num_params = count_parameters(model)
    logger.info(f"Training parameters {args}")
    logger.info(f"Total Parameter: \t{num_params:.1f}M")
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info(f"  Num steps = {len(test_loader)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    loss_fct = nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            y = y.squeeze().long()
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    all_preds = np.concatenate(all_preds, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    
    accuracy = simple_accuracy(all_preds, all_label)
    
    logger.info("\n")
    logger.info("Validation Results")
    logger.info(f"Global Steps: {global_step}")
    logger.info(f"Valid Loss: {eval_losses.avg:.5f}")
    logger.info(f"Valid Accuracy: {accuracy:.5f}")

    writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)
    writer.add_scalar("valid/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, eval_losses.avg

def test(args):
    eval_losses = AverageMeter()
    args.stage = 'test'
    args, model = setup(args)
    test_loader = get_loader(args)
    
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    loss_fct = nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            y = y.squeeze().long()
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        epoch_iterator.set_description(f"Testing... (loss={eval_losses.val:.5f})")

    all_preds = np.concatenate(all_preds, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    
    accuracy = simple_accuracy(all_preds, all_label)
    
    logger.info("\n")
    logger.info("Test Results")
    logger.info(f"Test Loss: {eval_losses.avg:.5f}")
    logger.info(f"Test Accuracy: {accuracy:.5f}")

def train(args, model):
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, val_loader = get_loader(args)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.num_steps}")
    logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    loss_fct = nn.CrossEntropyLoss()

    while global_step < args.num_steps:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            logits = model(x)
            y = y.squeeze().long()
            loss = loss_fct(logits, y)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    f"Training ({global_step} / {t_total} Steps) (loss={losses.val:.5f})"
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, val_loss = valid(args, model, writer, val_loader, global_step)
                    if accuracy > best_acc:
                        save_model(args, model, accuracy, args.mode)
                        best_acc = accuracy

                if global_step >= args.num_steps:
                    break
        losses.reset()
        if global_step >= args.num_steps:
            break

    writer.close()
    logger.info(f"Best Accuracy: \t{best_acc:.5f}")
    logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vit_tiny_patch16', 'vit_base_patch16', 'vit_large_patch16', 'vit_large_patch32'],
                        default='vit_base_patch16', type=str, help='Name of model to train')
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    parser.add_argument("--num_classes", default=2, type=int, help="Number of classes for COVIDx")
    parser.add_argument("--pretrained_path", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Path to pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int, help="Run validation every so many steps.")
    parser.add_argument("--learning_rate", default=3e-2, type=float, help="Initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if applied.")
    parser.add_argument("--num_steps", default=10000, type=int, help="Total number of training steps.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Steps for learning rate warmup.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Use 16-bit float precision.")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="Apex AMP optimization level: ['O0', 'O1', 'O2', 'O3'].")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling for fp16. 0 for dynamic scaling.")
    parser.add_argument("--dataset_path", type=str, help="Path to COVIDx dataset.")
    parser.add_argument("--mode", type=str, default="Finetune", choices=["Finetune", "LinearProbe"],
                        help="Finetune or LinearProbe mode.")

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}, "
                   f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    set_seed(args)
    args, model = setup(args)

    if args.stage == "train":
        train(args, model)
    else:
        test(args)

if __name__ == "__main__":
    main()

