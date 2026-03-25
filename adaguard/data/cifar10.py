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


def partition_data_non_iid(dataset, num_clients, num_classes=10,
                           classes_per_client=2, min_samples=20):
    """Partition dataset into non-IID client shards.

    Each client gets `classes_per_client` classes with samples drawn from those classes.
    Samples are shared across clients assigned to the same class pair (realistic FL).

    Args:
        dataset: CIFAR-10 dataset
        num_clients: number of FL clients
        num_classes: number of classes (10 for CIFAR-10)
        classes_per_client: how many classes each client sees (default 2 = non-IID)
        min_samples: minimum samples per client (pads with random sampling if needed)
    """
    label_indices = defaultdict(list)
    targets = dataset.targets if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]
    for idx, label in enumerate(targets):
        label_indices[label].append(idx)

    # Shuffle indices within each class
    for c in label_indices:
        random.shuffle(label_indices[c])

    client_data = defaultdict(list)
    client_classes = {}

    # Assign each client a RANDOM subset of classes (much more diverse than sequential)
    all_classes = list(range(num_classes))
    for cid in range(num_clients):
        assigned = tuple(sorted(random.sample(all_classes, classes_per_client)))
        client_classes[cid] = assigned

    # For each client, sample data from their assigned classes
    # Use Dirichlet-like allocation: each client gets a roughly equal share
    # of data from each of their assigned classes
    samples_per_class = {}  # track how many samples we've allocated per class
    for c in range(num_classes):
        samples_per_class[c] = 0

    # Count how many clients want each class
    class_demand = defaultdict(int)
    for cid, assigned in client_classes.items():
        for c in assigned:
            class_demand[c] += 1

    # Allocate: each client gets (class_size / demand) samples from each class
    for cid in range(num_clients):
        for c in client_classes[cid]:
            class_indices = label_indices[c]
            n_available = len(class_indices)
            demand = class_demand[c]
            # Each client wanting this class gets an equal share
            share = max(min_samples // classes_per_client, n_available // demand)
            start = (samples_per_class[c]) % n_available
            selected = []
            for i in range(share):
                selected.append(class_indices[(start + i) % n_available])
            client_data[cid].extend(selected)
            samples_per_class[c] += share

    # Ensure minimum samples
    for cid in range(num_clients):
        if len(client_data[cid]) < min_samples:
            assigned = client_classes[cid]
            all_class_idx = []
            for c in assigned:
                all_class_idx.extend(label_indices[c])
            while len(client_data[cid]) < min_samples:
                client_data[cid].append(random.choice(all_class_idx))

    for cid in client_data:
        random.shuffle(client_data[cid])

    # Summary
    sizes = [len(v) for v in client_data.values()]
    n_combos = len(set(client_classes.values()))
    print(f"  Partitioned into {num_clients} clients: "
          f"{min(sizes)}-{max(sizes)} samples/client (mean {sum(sizes)/len(sizes):.0f}), "
          f"{classes_per_client} classes/client, {n_combos} unique class combos")

    return dict(client_data)
