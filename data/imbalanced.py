import torchvision
import numpy as np

def num_exp_samples(idx, beta=100, n_classes=10, max_n_samples=5000):
    class_dist = max_n_samples * (1 / beta) ** (idx / (n_classes - 1))
    return class_dist.astype(np.int32)


def get_class_distribution_step(targets, config, beta):
    '''
    Calculate number of samples with each label
    for given beta. Includes functionality to combine, subset, and relabel
    classes.
    '''
    
    # Get old distribution of labels
    # Separate into list of head class and tail class sizes
    unique, counts = np.unique(targets, return_counts=True)
    head_class_sizes, tail_class_sizes = [], []
    for new_label, class_info in enumerate(config):
        label_count = 0
        for old_label in class_info['old_labels']:
            label_count += counts[np.where(unique == old_label)[0]][0]
        head_class_sizes.append(label_count) if class_info['class_type'] == 'head' \
            else tail_class_sizes.append(label_count)
    if beta is None:
        assert len(head_class_sizes) == 1 and len(tail_class_sizes) == 1, \
            f"beta=None only implemented for binary problems."
        n_tail = tail_class_sizes[0]
        n_head = head_class_sizes[0]
    else:
        # Find class distribution such that beta = head_class_size / tail_class_size
        # Head class size should be the minimum size of head classes
        if min(head_class_sizes) / min(tail_class_sizes) >= beta:
            n_tail = int(min(tail_class_sizes))
            n_head = int(beta * n_tail)
        else:
            n_head = int(min(head_class_sizes))
            n_tail = int(n_head / beta)
        
    # Get new class distribution and label map
    class_dist = [0] * len(config)
    label_map = {}
    for new_label, class_info in enumerate(config):      
        class_dist[new_label] = n_head if class_info['class_type'] == 'head' else n_tail
        label_map[new_label] = class_info['name']
    
    return class_dist, label_map
    

def get_class_dist_exp(targets, beta):
    '''
    Calculate number of samples with each label
    for given beta. Distribute number of samples exponentially with smallest labels \
    having the most samples. Assumes all classes are labeled 0 to N-1 where N is the \
    number of classes
    '''
    unique, counts = np.unique(targets, return_counts=True)
    assert min(unique) == 0, "minimum label value should be 0."
    assert max(unique) == len(unique) - 1, "maximum label value should be # classes - 1"
    n_classes = len(unique)
    return num_exp_samples(unique, beta=beta, n_classes=n_classes, 
                                    max_n_samples=max(counts))



def gen_imbalanced_data(data, targets, class_dist, imb_type='step', config=None, numpy_data=True):
    '''
    Get imbalanced version of dataset following specifications in config and
    class_dist.
    '''
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    for new_label, n_samples in enumerate(class_dist):
        if imb_type == 'step':
            idx = np.where(np.in1d(targets_np, config[new_label]['old_labels']))[0]
        elif imb_type == 'exp':
            idx = np.where(targets_np == new_label)[0]
        else:
            exit(f'Invalid imbalance type {imb_type}.')
        n_samples = class_dist[new_label]
        np.random.shuffle(idx)
        select_idx = idx[:n_samples]
        # If data is numpy array (like CIFAR data), can use indexing and then vstack.
        if numpy_data:
            new_data.append(data[select_idx, ...])
        else:
            for idx in select_idx:
                new_data.append(data[idx][0])
        new_targets.extend([new_label, ] * n_samples)
    if numpy_data:
        new_data = np.vstack(new_data)
        return new_data, new_targets
    else:
        new_samples = list(zip(new_data, new_targets))
        return new_samples, new_targets


class ImbalancedCifar(torchvision.datasets.CIFAR10):
    '''
    Class for creating an imbalanced set of CIFAR10 or CIFAR100 samples.
    Currently only supports step imbalance.
    Cite: https://github.com/val-iisc/Saddle-LongTail/blob/main/imbalance_cifar.py
    '''
    def __init__(self, dataset, beta, imb_type, config=None, rand_number=0):
        np.random.seed(rand_number)
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

        self.beta = beta
        self.config = config
        if imb_type == 'step':
            self.class_dist, self.label_map = get_class_distribution_step(dataset.targets, config, beta)
        elif imb_type == 'exp':
            self.class_dist = get_class_dist_exp(dataset.targets, beta)
            self.label_map = config
        self.data, self.targets = gen_imbalanced_data(dataset.data, dataset.targets, self.class_dist,
                                            imb_type=imb_type, config=config,
                                            numpy_data=True)
        self.n_classes = len(self.class_dist)


class ImbalancedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, dataset, beta, imb_type, config=None, rand_number=0):
        np.random.seed(rand_number)
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.loader = dataset.loader

        self.beta = beta
        self.config = config
        if imb_type == 'step':
            self.class_dist, self.label_map = get_class_distribution_step(dataset.targets, config, beta)
        elif imb_type == 'exp':
            self.class_dist = get_class_dist_exp(dataset.targets, beta)
            self.label_map = config
        self.samples, self.targets = gen_imbalanced_data(dataset.samples, 
                                                         dataset.targets, 
                                                         self.class_dist, 
                                                         imb_type=imb_type,
                                                         config=config,
                                                         numpy_data=False)
        self.n_classes = len(self.class_dist)

