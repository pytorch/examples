from torchvision import transforms


def transform_for_training():
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.Resize((96, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def transform_for_infer():
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.Resize((96, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )