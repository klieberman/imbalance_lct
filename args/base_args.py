import argparse
import yaml
import sys
import json

import data
import models
import optimizers


def int_or_none(value):
    if type(value) != 'int' and value.lower() == 'none':
        return None
    return int(value)


def float_or_none(value):
    if value.lower() == 'none':
        return None
    return float(value)  


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1
    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def get_arg_str(args):
    s = '------------ Options -------------\n'
    for k, v in args.items():
        s += "{}: {}\n".format(k, v)
    s += '-------------- End ----------------\n'
    return s


class BaseArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            'configs',
            nargs='*',
            default=None,
            help='Optional list of yaml config files with argument specifications.'
        )     
        
        # Data
        self.parser.add_argument(
            '--data_dir',
            default="../datasets",
            help="Directory of training dataset. \
                If using base_dataset is cifar10 or cifar100 and the dataset \
                is not in this folder, then it will be downloaded and stored here.\
                If using a different dataset, then data should be organized into folders \
                based on the class labels."
        )
        self.parser.add_argument(
            '--data_val_dir',
            default="../datasets",
            help="Directory of validation or test dataset. \
                If using base_dataset is cifar10 or cifar100 and the dataset \
                is not in this folder, then it will be downloaded and stored here.\
                If using a different dataset, then data should be organized into folders \
                based on the class labels."
        )
        self.parser.add_argument(
            '--base_dataset',
            default="cifar10",
            choices=data.available_base_datasets(),
            help="Name of original dataset which can then be subsetted \
                based on information in dataset_classes."
        )
        self.parser.add_argument(
            '--dataset_name',
            default="cat_dog",
            help="Name of dataset. Used for saving runs."
        )
        self.parser.add_argument(
            '--dataset_classes',
            default=[{'name': 'cat', 'class_type': 'head', 'old_labels': [3]},
                     {'name': 'dog', 'class_type': 'tail', 'old_labels': [5]}],
            help="Dictionary with information about classes. \
                Can be specified in a yml config file. \
                Each class should contain a name, class_type (head or tail), and \
                a list of old_labels or the labels associated with this class in \
                the base_dataset."
        )
        self.parser.add_argument(
            '--cifar10_pair',
            default=None,
            type=int,
            nargs=2,
            help="Option to specify a pair of cifar10 classes instead of specifying \
                  dataset_classes. Must be a list of two integers between 0-9 where \
                  the first entry is the label of the majority class and the second \
                  is the label of the minority class. If specified, base_dataset, \
                  dataset_name, and dataset_classes will be overridden to proper values."
        )
        self.parser.add_argument(
            '--beta',
            default=10,
            type=int_or_none,
            help="Ratio of head samples to tail samples in the train split.\
                If None, then the original data distribution will be used."
        )
        self.parser.add_argument(
            '--imb_type',
            default='step',
            choices=['step', 'exp'],
            help="Step or exponential imbalance."
        )

        # Model
        self.parser.add_argument(
            '--model',
            choices=models.available_models().keys(),
            default="resnet32",
            help="Name of model to use. If using ResNeXt model, the weights will be \
                initialized to ResNeXt50_32X4D_Weights.IMAGENET1K_V1."
        )
        self.parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to checkpoint to resume from in run_dir/checkpoints. \
            If none is specified at test time, then most_recent.state is used.",
        )

        # Loss function hyperparameters
        self.parser.add_argument(
            '--omega',
            type=float,
            nargs='+',
            default=[0.5],
            help="Omega in VS loss. Must be a scalar float or a list of 3 floats.\
                If 1 value is given, omega will be set to this constant. \
                If 3 values are given, omega will be part of the lambda vector for LCT \
                and these values will be the (a, b, h_b) on the linear distribution \
                omega is drawn from."
        )
        self.parser.add_argument(
            '--gamma',
            type=float,
            nargs='+',
            default=[0.],
            help="Gamma in VS loss. Must be a scalar float or a list of 3 floats.\
                If 1 value is given, gamma will be set to this constant. \
                If 3 values are given, gamma will be part of the lambda vector for LCT \
                and these values will be the (a, b, h_b) on the linear distribution \
                gamma is drawn from."
        )
        self.parser.add_argument(
            '--tau',
            type=float,
            nargs='+',
            default=[0.],
            help="Tau in VS loss. Must be a scalar float or a list of 3 floats.\
                If 1 value is given, tau will be set to this constant. \
                If 3 values are given, tau will be part of the lambda vector for LCT \
                and these values will be the (a, b, h_b) on the linear distribution \
                tau is drawn from."
        )
        

        self.parser.add_argument(
            '--optimizer',
            choices=optimizers.available_optimizers().keys(),
            default="sgd",
            help="Name of optimizer to use."
        )
        self.parser.add_argument(
            '--sam_rho',
            type=float,
            default=0.5,
            help='Rho used for sharpness aware minimization (SAM) optimizer.'
        )
        self.parser.add_argument(
            '--grad_max_norm',
            default=0.5,
            type=float_or_none,
            help='Max norm of gradient before clipping. Use None for no gradient clipping.'
        )
        self.parser.add_argument(
            '--epochs',
            default=500,
            type=int,
            help="Number of training epochs."
        )
        self.parser.add_argument(
            '--save_every',
            type=int,
            default=100,
            help='How many training epochs to save after.'
        )


        # Hardware, seeds, and EMA
        self.parser.add_argument(
            "--workers",
            default=2,
            type=int,
            help="Number of data loading workers.",
        )
        self.parser.add_argument(
            '--random_seed',
            default=23,
            help="Random seed for numpy."
        )
        self.parser.add_argument(
            '--ema_decay',
            type=float,
            default=0.9,
            help='Exponential Movaing Average (EMA) decay.'
        )
        self.parser.add_argument(
            '--eval_with_ema',
            default=True,
            action=argparse.BooleanOptionalAction,
            help='Whether to evaluate with EMA.'
        )
        self.parser.add_argument(
            '--show_tqdm',
            default=False,
            action=argparse.BooleanOptionalAction,
            help='Whether to show the tqdm outputs.'
        )


    def parse(self):
        if not self.initialized:
            self.initialize()

        args = self.parser.parse_args()

        for config in args.configs:
            if config is not None:
                # get commands from command line
                override_args = argv_to_vars(sys.argv)

                # load yaml file
                yaml_txt = open(config).read()
                loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
                for k, v in loaded_yaml.items():
                    if type(v) == "dictionary":
                        loaded_yaml[k]=json.dumps(v)

                # load commmand line arguments (overriding any arguments in the yaml file)
                for v in override_args:
                    # handle false case of booleanoptionalaction
                    if v.startswith('no_'):
                        loaded_yaml[v[3:]] = False
                    else:
                        loaded_yaml[v] = getattr(args, v)

                print(f"=> Reading YAML config from {config}")
                args.__dict__.update(loaded_yaml)

        self.args = args
        self.arg_str = get_arg_str(vars(args))
        print(self.arg_str)

        return args


    def save(self, file_name):
        with open(file_name, 'wt') as opt_file:
            opt_file.write(self.arg_str)