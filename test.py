import numpy as np 
import os.path as osp

from args.test_args import TestArguments
from trainer import Trainer



if __name__ == '__main__':
    test_args = TestArguments()
    all_args = test_args.parse()
    
    trainer = Trainer(all_args, train=False)

    # Evaluate model and save accuracies, ys, and y_hats
    if trainer.lct:
        acc_meter, full_y, full_y_hat = trainer.eval_model(
            "test", 
            eval_lambda_omega=all_args.eval_lambda_omega,
            eval_lambda_gamma=all_args.eval_lambda_gamma,
            eval_lambda_tau=all_args.eval_lambda_tau
            )
    else:
        acc_meter, full_y, full_y_hat = trainer.eval_model("test")
    results = {
        'overall_acc': acc_meter.overall_acc1,
        'tnr': acc_meter.head_acc1,
        'tpr': acc_meter.tail_acc1,
        'fpr': acc_meter.fpr,
        'full_y': full_y,
        'full_y_hat': full_y_hat
    }
    
    # Save statistics to .json file
    if trainer.lct:
        results_file = osp.join(
            trainer.results_dir,
            f'epoch_{trainer.epoch}_{all_args.eval_lambda_omega}_' \
            f'{all_args.eval_lambda_gamma}_{all_args.eval_lambda_tau}.npy'
            )
    else:
        results_file = osp.join(trainer.results_dir, f'epoch_{trainer.epoch}.npy')
    np.save(results_file, results)
    print(f"Saved results to {results_file}")

    # To read this, use:
    # dict = np.load(osp.join(results_dir, "results.npy"), allow_pickle=True)
    # dict.item().get('roc')
