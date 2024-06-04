
# Cite: bpep repository
import abc
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, tqdm_writer=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if not tqdm_writer:
            print("\t".join(entries))
        else:
            tqdm.tqdm.write("\t".join(entries))

    def write_to_tensorboard(
        self, writer: SummaryWriter, prefix="train", global_step=None
    ):
        for meter in self.meters:
            avg = meter.avg
            val = meter.val
            if meter.write_val:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_val", val, global_step=global_step
                )

            if meter.write_avg:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_avg", avg, global_step=global_step
                )

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class AverageMeter(Meter):
    """
    Computes and stores the average value.
    """

    def __init__(self, name, write_val=True, write_avg=True):
        self.name = name
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

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

    def __str__(self):
        display_str = f"{self.name}: "
        if self.write_val:
            display_str += f"{self.val:.3f}"
        if self.write_avg:
            display_str += f"{self.avg:.3f}"
        return display_str
    
    def display(self):
        return self.__str__()
    
    def write_to_tensorboard(self, writer: SummaryWriter, prefix="train", global_step=None):
        writer.add_scalar(f"{prefix}/{self.name}_avg", self.avg, global_step=global_step)
    

def true_false_pos_neg(y, y_hat):
    y_pred = torch.argmax(y_hat, dim=1)
    tp = torch.sum(torch.where((y==1) & (y_pred == 1), 1, 0)).item()
    fp = torch.sum(torch.where((y==0) & (y_pred == 1), 1, 0)).item()
    fn = torch.sum(torch.where((y==1) & (y_pred == 0), 1, 0)).item()
    tn = torch.sum(torch.where((y==0) & (y_pred == 0), 1, 0)).item()
    return tp, fp, fn, tn


def tail_head_correct(y, y_hat, tail_classes):
    y_pred = torch.argmax(y_hat, dim=1)
    tail_correct = torch.sum(torch.where(torch.isin(y, tail_classes) & (y_pred == y), 1, 0)).item()
    tail_incorrect = torch.sum(torch.where(torch.isin(y, tail_classes) & (y_pred != y), 1, 0)).item()
    head_correct = torch.sum(torch.where(torch.isin(y, tail_classes, invert=True) & (y_pred == y), 1, 0)).item()
    head_incorrect = torch.sum(torch.where(torch.isin(y, tail_classes, invert=True) & (y_pred != y), 1, 0)).item()
    return tail_correct, tail_incorrect, head_correct, head_incorrect
    

class AccuracyMeter(Meter):
    def __init__(self, n_samples, tail_classes, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(n_samples)
        self.prefix = prefix
        self.tail_classes = tail_classes
        self.reset()

    def reset(self):
        self.correct_tail = 0
        self.incorrect_tail = 0
        self.correct_head = 0
        self.incorrect_head = 0
        self.total = 0

        self.tail_acc1 = None
        self.head_acc1 = None
        self.overall_acc1 = None
        self.fpr = None


    def update(self, y, y_hat):
        correct_tail, incorrect_tail, correct_head, incorrect_head = tail_head_correct(y, y_hat, self.tail_classes)
        self.total = self.total + correct_tail + incorrect_tail + correct_head + incorrect_head
        self.correct_tail += correct_tail
        self.incorrect_tail += incorrect_tail
        self.correct_head += correct_head
        self.incorrect_head += incorrect_head
        
        if self.correct_tail + self.incorrect_tail:
            self.tail_acc1 = self.correct_tail / (self.correct_tail + self.incorrect_tail)
        if self.correct_head + self.incorrect_head:
            self.head_acc1 = self.correct_head / (self.correct_head + self.incorrect_head)
        self.overall_acc1 = (self.correct_tail + self.correct_head) / (self.total)
        if self.incorrect_head + self.correct_head:
            # fpr = fp / (fp + tn)
            self.fpr = self.incorrect_head / (self.incorrect_head + self.correct_head)


    def write_to_tensorboard(
        self, writer: SummaryWriter, prefix="train", global_step=None
    ):
        writer.add_scalar(f"{prefix}/overall_acc1", self.overall_acc1, global_step=global_step)
        writer.add_scalar(f"{prefix}/tail_acc1", self.tail_acc1, global_step=global_step)
        writer.add_scalar(f"{prefix}/head_acc1", self.head_acc1, global_step=global_step)


    def display(self, tqdm_writer=True):
        entries = [self.prefix + self.batch_fmtstr.format(self.total)]
        entries += [f"overall_acc1: {self.overall_acc1:.3f}",
                    f"tail_acc1: {self.tail_acc1:.3f}",
                    f"head_acc1: {self.head_acc1:.3f}",]
        if not tqdm_writer:
            print("\t".join(entries))
        else:
            tqdm.tqdm.write("\t".join(entries))

    def _get_batch_fmtstr(self, n_steps):
        num_digits = len(str(n_steps // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(n_steps) + "]"
    