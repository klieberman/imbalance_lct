import argparse 

from args.base_args import BaseArguments


class TrainArguments(BaseArguments):
    def initialize(self):

        # Train dataset details
        self.parser.add_argument(
            '--data_seed', 
            default=0,
            type=int,
            help='Random seed for creating imbalanced dataset.'
        )
        self.parser.add_argument(
            '--validation_split',
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Whether to split train set into training and validation split. \
                If false, test set is used as validation set."
        )
        self.parser.add_argument(
            '--validation_prop',
            default=0.2,
            type=float,
            help="What proportion of the training data should be used for validation. \
                Only considered if args.validation_split is True."
        )
        self.parser.add_argument(
            '--test_train_split',
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Whether to test the train split at each epoch throughout training."
        )

        # Optimization parameters
        self.parser.add_argument(
            '--init_lr',
            default=0.1,
            type=float,
            help="Initial learning rate for weight training optimizer."
        )
        self.parser.add_argument(
            '--lr_steps',
            nargs = '*',
            default = [400, 1e-2, 450, 1e-3],
            help="Sequence of epochs where learning rate changes and associated \
                learning rate values. Sequence has format \
                [first_step, first_lr, second_step, second_lr, ...] for an \
                arbitrary number of steps. Default is that learning rate will init_lr \
                until epoch 400, 1e-2 for epochs 400-450, and 1e-3 for epochs 450-500."
        )
        self.parser.add_argument(
            '--warmup_epochs',
            default=5,
            type=int,
            help ="Number of epochs with warm-up (smaller) learning rates."
        )
        self.parser.add_argument(
             '--momentum',
             default=0.9,
             type=float,
             help="Momentum on optimizer."
        )
        self.parser.add_argument(
            '--weight_decay',
            type=float,
            default=2e-4,
            help='Weight decay.'
        )
        self.parser.add_argument(
            '--skip_existing',
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Whether to skip training if a checkpoint which matches the arguments \
                already exists."
        )

        # Batches and epochs
        self.parser.add_argument(
            "--batch_size",
            default=128,
            type=int,
            help="Mini-batch size.",
        )
        BaseArguments.initialize(self)

