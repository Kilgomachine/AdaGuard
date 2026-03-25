"""CIFAR-10 data loading and non-IID client partitioning."""

import random
from collections import defaultdict

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_cifar10(data_root='./data'):
    """Load CIFAR-10 train and test datasets with standard normalization.

    Training set uses data augmentation (random crop + horizontal flip) which is
    essential for FL with small per-client datasets to prevent overfitting.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])
    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform,
    )
    test_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform,
    )
    print(f"CIFAR-10 loaded — Train: {len(train_ds)}, Test: {len(test_ds)}")
    return train_ds, test_ds


def partition_data_non_iid(dataset, num_clients, num_classes=10):
    """Partition dataset into non-IID client shards (each client gets a subset of classes)."""
    label_indices = defaultdict(list)
    # Use .targets directly (fast) instead of iterating dataset[idx] (slow)
    targets = dataset.targets if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]
    for idx, label in enumerate(targets):
        label_indices[label].append(idx)

    client_data = defaultdict(list)
    classes_per_client = max(2, num_classes // num_clients)

    for cid in range(num_clients):
        assigned = [
            (cid * classes_per_client + j) % num_classes
            for j in range(classes_per_client)
        ]
        for c in assigned:
            indices = label_indices[c]
            s = (cid * len(indices)) // num_clients
            e = ((cid + 1) * len(indices)) // num_clients
            client_data[cid].extend(indices[s:e])

    for cid in client_data:
        random.shuffle(client_data[cid])
        print(f"  Client {cid}: {len(client_data[cid])} samples")

    return dict(client_data)
