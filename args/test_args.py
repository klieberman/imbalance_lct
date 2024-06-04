from args.base_args import BaseArguments
import argparse

class TestArguments(BaseArguments):
    def initialize(self):
        # Data information
        self.parser.add_argument(
            "--batch_size",
            default=1000,
            type=int,
            help="mini-batch size.",
        )
        self.parser.add_argument(
            '--eval_lambda_omega',
            default=0.5,
            type=float,
            help="Omega value to use for inference if omega is in LCT's lambda."
        )
        self.parser.add_argument(
            '--eval_lambda_gamma',
            default=0.0,
            type=float,
            help="Gamma value to use for inference if gamma is in LCT's lambda."
        )
        self.parser.add_argument(
            '--eval_lambda_tau',
            default=3.0,
            type=float,
            help="Tau value to use for inference if tau is in LCT's lambda."
        )


        BaseArguments.initialize(self)
        