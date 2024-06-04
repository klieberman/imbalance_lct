import time
import math
import numpy as np
from datetime import datetime
import os.path as osp

from args.train_args import TrainArguments
from utils.logging import AverageMeter, ProgressMeter
from trainer import Trainer


if __name__ == '__main__':
    train_args = TrainArguments()
    all_args = train_args.parse()
    trainer = Trainer(all_args, train=True)
    train_args.save(osp.join(trainer.config_dir, f"train_{trainer.dt_string}.txt"))

    # Train
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    test_train_time = AverageMeter("test_train_set_time", ":.4f", write_avg=False)
    val_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
       all_args.epochs, [epoch_time, train_time, test_train_time, val_time], 
       prefix="Overall Timing")
    print(f'lct?: {trainer.lct}')
    for epoch in range(trainer.epoch + 1, all_args.epochs + 1):
        start_epoch = time.time()

        # Train epoch
        print(f'Training epoch {epoch}...')
        start_train = time.time()
        trainer.train_epoch()
        train_time.update((time.time() - start_train))

        if all_args.test_train_split:
            # Test train set
            print("Testing train set...")
            start_test_train = time.time()
            trainer.eval_model("train")
            test_train_time.update((time.time() - start_test_train))

        # Test validation set
        print("Testing validation set...")
        start_val = time.time()
        acc_meter, _, _ = trainer.eval_model("val")
        val_time.update((time.time() - start_val))

        # Save if validation set achieves best tail_acc1 or head_acc1
        if acc_meter.tail_acc1 > trainer.best_tail_acc1:
            trainer.best_tail_acc1 = acc_meter.tail_acc1
            trainer.save_checkpoint("best_tail_acc1.state")
        if acc_meter.head_acc1 > trainer.best_head_acc1:
            trainer.best_head_acc1 = acc_meter.head_acc1
            trainer.save_checkpoint("best_head_acc1.state")
        if acc_meter.overall_acc1 > trainer.best_overall_acc1:
            trainer.best_overall_acc1 = acc_meter.overall_acc1
            trainer.save_checkpoint("best_overall_acc1.state")
        if epoch % all_args.save_every == 0:
            trainer.save_checkpoint(f"epoch_{trainer.epoch:04d}.state")

        epoch_time.update((time.time() - start_epoch))
        progress_overall.display(epoch + 1)
        progress_overall.write_to_tensorboard(
            trainer.writer, prefix="z_time", global_step=epoch
        )
    
    # Save final model
    trainer.save_checkpoint(f"epoch_{trainer.epoch:04d}.state")

