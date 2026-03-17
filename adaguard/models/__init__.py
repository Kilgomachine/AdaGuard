from .cnn import SmallCNN
from .resnet import ResNet18, ResNet34, ResNet50

MODEL_REGISTRY = {
    'smallcnn': SmallCNN,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
}


def create_model(name='smallcnn', num_classes=10):
    """Create a model by name. Supported: smallcnn, resnet18, resnet34, resnet50."""
    factory = MODEL_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return factory(num_classes=num_classes)
