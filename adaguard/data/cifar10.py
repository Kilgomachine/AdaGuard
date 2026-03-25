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

    # Group clients by their class assignments
    # Client cid gets classes: {(cid * classes_per_client + j) % num_classes}
    # Multiple clients share the same class pair — split that class's data among them
    class_pair_clients = defaultdict(list)  # class_pair -> list of client_ids
    client_classes = {}  # cid -> list of assigned classes

    for cid in range(num_clients):
        assigned = tuple(sorted([
            (cid * classes_per_client + j) % num_classes
            for j in range(classes_per_client)
        ]))
        client_classes[cid] = assigned
        class_pair_clients[assigned].append(cid)

    # For each class pair, distribute that pair's data evenly among clients
    for pair, cids in class_pair_clients.items():
        # Collect all indices for this class pair
        pair_indices = []
        for c in pair:
            pair_indices.extend(label_indices[c])
        random.shuffle(pair_indices)

        # Split evenly among clients in this pair
        n = len(cids)
        chunk_size = len(pair_indices) // n
        for i, cid in enumerate(cids):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else len(pair_indices)
            client_data[cid].extend(pair_indices[start:end])

    # Ensure minimum samples (pad with random sampling from assigned classes)
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
    n_pairs = len(class_pair_clients)
    print(f"  Partitioned into {num_clients} clients: "
          f"{min(sizes)}-{max(sizes)} samples/client (mean {sum(sizes)/len(sizes):.0f}), "
          f"{classes_per_client} classes/client, {n_pairs} unique class pairs")

    return dict(client_data)
