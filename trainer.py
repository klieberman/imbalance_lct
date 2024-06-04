import os.path as osp
import numpy as np
import torch
import torch.nn
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from torchvision.models.resnet import ResNeXt50_32X4D_Weights
from datetime import datetime

from data import available_base_datasets
from data.cifar10_info import LABEL_MAP as CIFAR10_LABEL_MAP
from data.utils import split_evenly, print_class_distribution, print_dataset_size
from models import available_models, lct_models
from models.lct_layers import FiLMLinearBlock
from losses import VSLoss, VSLossLCT
from optimizers import available_optimizers
from utils.lct import get_lambdas_all
from utils.logging import AverageMeter, AccuracyMeter
from utils.utilities import get_device, makedirs_if_needed, print_model_param_nums

class Trainer:
    '''
    Trainer which manages the datasets, model, loss criterion, and optimizer.
    '''
    def __init__(self, args, train=True):
        self.train = train
        self.device = get_device()
        np.random.seed(args.random_seed)
        now = datetime.now()
        self.dt_string = now.strftime("%y_%m_%d:%H:%M")
        self.grad_max_norm = args.grad_max_norm
        self.show_tqdm = args.show_tqdm

        # If using CIAR10 pair, set relevant args for dataset
        if args.cifar10_pair is not None:
            class_0, class_1 = args.cifar10_pair
            assert class_0 in range(10) and class_1 in range(10), "CIFAR10 labels must" \
                "be between 0-9."
            args.base_dataset = 'cifar10'
            args.dataset_name = f"{class_0}_{class_1}"
            args.dataset_classes = [
                {'name': CIFAR10_LABEL_MAP[class_0], 'class_type': 'head', 'old_labels': [class_0]},
                {'name': CIFAR10_LABEL_MAP[class_1], 'class_type': 'tail', 'old_labels': [class_1]}
            ]
        
        # Determine which hyperparams are in the lambda vector for
        # Loss Conditional Training (LCT)
        self.lambda_vars = {}
        for name, info in zip(['omega', 'gamma', 'tau'],
                              [args.omega, args.gamma, args.tau]):
            assert len(info) in [1, 3], f"args.{name} must contain 1 or 3 values."
            if info is not None and len(info) == 3:
                self.lambda_vars[name] = {'dist': 'linear', \
                                          'a': info[0], 'b': info[1], 'hb': info[2]}
        self.omega = args.omega[0] if len(args.omega) == 1 else args.omega
        self.gamma = args.gamma[0] if len(args.gamma) == 1 else args.gamma
        self.tau = args.tau[0] if len(args.tau) == 1 else args.tau
        self.n_lambdas = len(self.lambda_vars)

        # Get relevant directories, save config file, and prepare for logging
        self.lct = len(self.lambda_vars) > 0
        hyper_params = f'omega_{args.omega}_gamma_{args.gamma}_tau_{args.tau}/'        
        if self.lct:
            assert args.model in lct_models(), "Hyperparameter arguments suggest that" \
                " LCT should be used, but model is not a LCT model."
        if args.optimizer == "sam": 
            hyper_params += f"{args.optimizer}_{args.sam_rho}"
        else:
            hyper_params += f"{args.optimizer}"
        hyper_params += f"_{args.epochs}"
        self.run_dir = osp.join(f"runs/{args.dataset_name}{args.beta}_clip_"\
                                f"{args.grad_max_norm}/{args.model}", hyper_params)
        if train and args.resume is None and osp.exists(self.run_dir):
            if args.skip_existing and osp.exists(f"{self.run_dir}/checkpoints/epoch_"\
                                                 f"{args.epochs:04d}.state"):
                exit('Results already exist, skipping.')
            else:
                self.run_dir=f'{self.run_dir}/{self.dt_string}'
        elif not train and not osp.exists(self.run_dir):
            exit(f'run_dir {self.run_dir} does not exist.')
        print(f"Saving/testing checkpoints, logs, and results in {self.run_dir}.\n")
        self.log_dir = makedirs_if_needed(osp.join(self.run_dir, "logs"))
        self.checkpoint_dir = makedirs_if_needed(osp.join(self.run_dir, "checkpoints"))
        self.results_dir = makedirs_if_needed(osp.join(self.run_dir, "results"))
        self.config_dir = makedirs_if_needed(osp.join(self.run_dir, "configs"))
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Prepare datasets and loaders
        data_info = available_base_datasets()[args.base_dataset]
        if args.base_dataset in ['cifar10', 'cifar100']:
            self.dataset_kwargs = {'train': True, 'download': True}
        else:
            self.dataset_kwargs = {}
        self.loader_kwargs = {"num_workers": args.workers, "pin_memory": True}
        if train:
            train_dataset = data_info.CLASS(root=args.data_dir,
                            transform=data_info.TRAIN_TRANSFORM,
                            **self.dataset_kwargs)

            if args.validation_split:
                train_dataset, val_dataset = split_evenly(train_dataset, 
                                                        args.validation_prop)
            else:
                if args.base_dataset in ['cifar10', 'cifar100']:
                    self.dataset_kwargs['train'] = False
                val_dataset = data_info.CLASS(root=args.data_val_dir,
                                                transform=data_info.VAL_TRANSFORM,
                                                **self.dataset_kwargs)
            
            # Get imbalanced/relabeled train set
            train_dataset = data_info.IMBALANCED_CLASS(train_dataset,
                                                beta=args.beta,
                                                imb_type=args.imb_type,
                                                config=args.dataset_classes if args.imb_type=='step' else data_info.LABEL_MAP,
                                                rand_number=args.data_seed,
                                                )
            # Relabel validation set
            val_dataset = data_info.IMBALANCED_CLASS(val_dataset,
                                        beta=None if args.imb_type == 'step' else 1,
                                        imb_type=args.imb_type,
                                        config=args.dataset_classes if args.imb_type=='step' else data_info.LABEL_MAP
                                        )
            print_class_distribution(train_dataset, val_dataset=val_dataset)
            
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=None,
                shuffle=True,
                **self.loader_kwargs
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                **self.loader_kwargs
            )

            self.train_set_size = len(train_dataset)
            self.val_set_size = len(val_dataset)
            print_dataset_size('train', train_dataset, self.train_loader, 
                               args.batch_size)
            print_dataset_size('validation', val_dataset, self.val_loader, 
                               args.batch_size)
            self.n_classes = len(train_dataset.class_dist)

        else:
            if args.base_dataset in ['cifar10', 'cifar100']:
                self.dataset_kwargs = {'train': False}      
            test_dataset = data_info.CLASS(root=args.data_val_dir,
                                                transform=data_info.VAL_TRANSFORM,
                                                **self.dataset_kwargs)
            test_dataset = data_info.IMBALANCED_CLASS(test_dataset,
                                                      beta=None if args.imb_type == 'step' else 1,
                                                      imb_type=args.imb_type,
                                                      config=args.dataset_classes if args.imb_type == 'step' else data_info.LABEL_MAP
                                                      )
            self.test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        **self.loader_kwargs
                    )
            self.test_set_size = len(test_dataset)
            print_dataset_size('test', test_dataset, self.test_loader, args.batch_size)
            print_class_distribution(test_dataset)
            self.n_classes = test_dataset.n_classes
        
        if args.imb_type == 'step':
            tail_classes = []
            for label, class_info in enumerate(args.dataset_classes):
                if class_info['class_type'] == 'tail':
                    tail_classes.append(label)
        else:
            # 30% of the smallest classes
            tail_classes = np.arange(int(0.7 * (self.n_classes)), self.n_classes - 1)
        self.tail_classes = torch.Tensor(tail_classes).to(self.device)

        # Initialize model
        if args.model in ['resNext50-32x4d', 'resNext50-32x4d_lct']:
            self.model = available_models()[args.model](
                weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.n_classes)
            if self.lct:
                self.model.film_layer = FiLMLinearBlock(self.n_lambdas, self.n_classes)
        else:
            if self.lct:
                self.model = available_models()[args.model](num_classes=self.n_classes, 
                                                            n_lambdas=self.n_lambdas)
            else:
                self.model = available_models()[args.model](num_classes=self.n_classes)
        print_model_param_nums(self.model)
        self.model = self.model.to(self.device)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=args.ema_decay)
        self.eval_with_ema = args.eval_with_ema
        self.total_epochs = args.epochs

        # Prepare optimizers and lr schedulers
        if train:
            self._set_criterion()
            self.optimizer_type = args.optimizer
            self._set_optimizer(args.optimizer, args.init_lr, args.momentum, 
                                args.weight_decay, args.sam_rho)
            if args.lr_steps is None:
                self.scheduler_info = None
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_epochs)
            else:
                self.scheduler_info = {
                    'init_lr': args.init_lr, 
                    'warmup_epochs': args.warmup_epochs,
                    'lr_steps': args.lr_steps
                }
        else:
            self.optimizer_type = self.optimizer = self.scheduler_info = None

        # If a particular resume file is specified, resume from there
        if args.resume is not None:
            self.resume(args.resume)
        # If no file is specified and train is False, resume from most_recent.state
        elif not train:
            self.resume('most_recent.state')
        # Otherwise, train from scratch
        else:
            self.epoch = 0
            self.best_tail_acc1 = 0
            self.best_head_acc1 = 0
            self.best_overall_acc1 = 0


    def resume(self, resume_file):
        full_resume_file = osp.join(self.checkpoint_dir, resume_file)
        assert osp.isfile(full_resume_file), f"{full_resume_file} is not a file"
        print(f"Resuming from {full_resume_file}.\n")
        checkpoint = torch.load(full_resume_file, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.best_tail_acc1 = checkpoint['best_tail_acc1']
        self.best_head_acc1 = checkpoint['best_head_acc1']
        if 'best_overall_acc1' in checkpoint:
            self.best_overall_acc1 = checkpoint["best_overall_acc1"]
        else:
            self.best_overall_acc1 = 0
        self.model.load_state_dict(checkpoint["state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])

        if self.optimizer is not None and checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print(f"=> Loaded checkpoint '{full_resume_file}' (epoch={self.epoch})")
        return


    def save_checkpoint(self, filename):
        state = {
            'epoch': self.epoch,
            'best_tail_acc1': self.best_tail_acc1,
            'best_head_acc1': self.best_head_acc1,
            'best_overall_acc1': self.best_overall_acc1,
            'state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict()
        }

        if self.optimizer is None:
            state['optimizer'] = None
        else:
            state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, osp.join(self.checkpoint_dir, filename))
        print(f"Saved checkpoint to {osp.join(self.checkpoint_dir, filename)}")
        return
    

    def _adjust_learning_rate(self):
        if self.scheduler_info is None:
            if self.epoch > 0:
                self.lr_scheduler.step()
        else:
            if self.epoch < self.scheduler_info['warmup_epochs']:
                self.lr = self.scheduler_info['init_lr'] \
                    * (self.epoch + 1) / self.scheduler_info['warmup_epochs']
            elif len(self.scheduler_info['lr_steps']) == 0 or \
                self.epoch < self.scheduler_info['lr_steps'][0]:
                self.lr = self.scheduler_info['init_lr']
            else:
                i = 0
                lr = self.scheduler_info['lr_steps'][i+1]
                while i < len(self.scheduler_info['lr_steps']) - 1 and self.epoch >= self.scheduler_info['lr_steps'][i]:
                    lr = self.scheduler_info['lr_steps'][i+1]
                    i += 2
                self.lr = lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        return
        

    def _set_criterion(self):
        if self.lct:
            self.criterion = VSLossLCT(
                class_dist=self.train_loader.dataset.class_dist,
                device=self.device,
                omega=self.omega if isinstance(self.omega, float) else None,
                gamma=self.gamma if isinstance(self.gamma, float) else None,
                tau=self.tau if isinstance(self.tau, float) else None,
            )
        else:
            self.criterion = VSLoss(
                class_dist=self.train_loader.dataset.class_dist,
                device=self.device,
                omega=self.omega,
                gamma=self.gamma,
                tau=self.tau
            )
    
    def _set_optimizer(self, optimizer, init_lr, momentum, weight_decay, sam_rho):
        if optimizer == "sam":
            optimizer_kwargs= {
                'base_optimizer': torch.optim.SGD,
                'rho': sam_rho
            }
        else:
            optimizer_kwargs = {}

        self.optimizer = available_optimizers()[optimizer](
            params=self.model.parameters(), lr=init_lr, momentum=momentum,
            weight_decay=weight_decay, **optimizer_kwargs)
        if optimizer == 'sam':
            for param_group in self.optimizer.param_groups:
                param_group['rho'] = sam_rho


    def train_epoch(self):
        
        self.model.train()
        self._adjust_learning_rate()
        num_batches = len(self.train_loader)
        if self.lct:
            epoch_lambdas = get_lambdas_all(self.lambda_vars, num_batches).to(self.device)

        for i, (x, y) in enumerate(tqdm.tqdm(self.train_loader, ascii=True, 
                                            total=num_batches, 
                                            disable=not self.show_tqdm)):
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = x.shape[0]

            # Forward pass
            if self.lct:
                batch_lambdas = epoch_lambdas[i].expand((batch_size, self.n_lambdas))
                y_hat = self.model(x, batch_lambdas)
                loss = self.criterion(y_hat, y, epoch_lambdas[i])
            else:
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_max_norm is not None:
                # Prevents exploding gradient by clipping gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.grad_max_norm, 
                                            error_if_nonfinite=True)

            if self.optimizer_type == "sam":
                self.optimizer.first_step(zero_grad=True)
                if self.lct:
                    batch_lambdas = epoch_lambdas[i].expand((batch_size, self.n_lambdas))
                    y_hat = self.model(x, batch_lambdas)
                    loss = self.criterion(y_hat, y, epoch_lambdas[i])
                else:
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()
        
        self.ema.update()

        self.epoch += 1
        # Always save most recent checkpoint
        self.save_checkpoint("most_recent.state")
        return
    

    def _test_split(self, loader, n_samples, split_name, eval_lambda_omega, 
                    eval_lambda_gamma, eval_lambda_tau):
        '''
        Tests the model on a particular data loader.
        Tracks the loss (if self.criterion is not none), accuracy, tpr, tnr, and fpr.
        Writes these values to tensorboard log file during training.
        Returns full tensor of y and y_hat during testing.
        '''
        n_batches = len(loader)
        acc_meter = AccuracyMeter(n_samples, prefix=split_name, 
                                  tail_classes=self.tail_classes)
        if hasattr(self, 'criterion'):
            loss_meter = AverageMeter("loss", write_val=False)

        if self.train:
            full_y_hat = full_y = None
        else:
            full_y = torch.zeros((n_samples))
            full_y_hat = torch.zeros((n_samples, self.n_classes))
            
        
        if self.lct:
            eval_lambdas = []
            for key, eval_lambda in zip(
                ['omega', 'gamma', 'tau'],
                [eval_lambda_omega, eval_lambda_gamma, eval_lambda_tau]
                ):
                if key in self.lambda_vars.keys():
                    eval_lambdas.append(eval_lambda)
            eval_lambdas = torch.tensor(eval_lambdas, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            i = 0
            for (x, y) in tqdm.tqdm(loader, ascii=True, total=n_batches, 
                                    disable=not self.show_tqdm):
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)

                if self.lct:
                    batch_lambdas = eval_lambdas.expand((batch_size, self.n_lambdas))

                if self.eval_with_ema:
                    with self.ema.average_parameters():
                        if self.lct:
                            y_hat = self.model(x, batch_lambdas)
                        else:
                            y_hat = self.model(x)
                else:
                    if self.lct:
                        y_hat = self.model(x, batch_lambdas)
                    else:
                        y_hat = self.model(x)

                # Record loss and accuracies
                if hasattr(self, 'criterion'):
                    if self.lct:
                        loss = self.criterion(y_hat, y, eval_lambdas)
                    else:
                        loss = self.criterion(y_hat, y)
                    loss_meter.update(loss.item(), batch_size)
                acc_meter.update(y, y_hat)
                
                if not self.train:
                    full_y[i: i + batch_size] = y.detach()
                    full_y_hat[i: i + batch_size] = y_hat.detach()
                i += batch_size

        acc_meter.display()
        if self.train:
            acc_meter.write_to_tensorboard(self.writer, prefix=split_name, 
                                           global_step=self.epoch)

        if hasattr(self, 'criterion'):
            print(str(loss_meter))
            loss_meter.write_to_tensorboard(self.writer, prefix=split_name, 
                                            global_step=self.epoch)

        return acc_meter, full_y, full_y_hat
    

    def eval_model(self, split, eval_lambda_omega=0.5, eval_lambda_gamma=0.0, 
                   eval_lambda_tau=3.0):
        '''
        Tests the model on the data in the given split 
        '''
        eval_lambdas = {
             'eval_lambda_omega': eval_lambda_omega,
             'eval_lambda_gamma': eval_lambda_gamma,
             'eval_lambda_tau': eval_lambda_tau,
        }
        if split == "train":
            return self._test_split(self.train_loader, self.train_set_size, 'train',
                                    **eval_lambdas)
        elif split == "val":
            return self._test_split(self.val_loader, self.val_set_size, 'val',
                                    **eval_lambdas)
        else:
            return self._test_split(self.test_loader, self.test_set_size, 'test',
                                    **eval_lambdas)