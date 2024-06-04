from data import cifar10_info, cifar100_info, petfinder_info, melanoma_info

_available_datasets = {
    'cifar10': cifar10_info,
    'cifar100': cifar100_info,
    'petfinder': petfinder_info,
    'melanoma': melanoma_info,
}


def available_base_datasets():
    return _available_datasets
