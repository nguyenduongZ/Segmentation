import cv2
import torch
import torchvision
import torchvision.datasets
import albumentations as A

from torch.utils.data import DataLoader, random_split

_root = "/media/mountHDD3/data_storage"

class OxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(
            self,
            root:str,
            split:str,
            target_types = "segmetation",
            download = False,
            transform = None,
            target_transform = None
    ):
        super().__init__(
            root = _root,
            split = split,
            target_types = target_types,
            download = download,
            transform = transform,
            target_transform = target_transform
        )
        self.transform = transform
        self.target_transform = transform

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        (img, msk) = super().__getitem__(idx)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            msk = self.target_transform(msk)
        return (img, msk)
    
transform = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)

target_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)

def get_ds(args):
    train_data = OxfordIIITPet(
        root = _root,
        split = "trainval",
        target_types = "segmentation",
        download = False,
        transform = transform
    )

    test_data = OxfordIIITPet(
        root = _root,
        split = "test",
        target_types = "segmentation",
        download = False,
        target_transform = target_transform
    )

    val_size = int(len(train_data) * args.val_split)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_dl = DataLoader(train_data, batch_size=args.bs, shuffle=True, pin_memory = arg.pm, num_workers=args.wk)

    val_dl = DataLoader(val_data, batch_size=args.bs, shuffle=True, pin_memory = arg.pm, num_workers=args.wk)

    test_dl = DataLoader(test_data, batch_size=args.bs, shuffle=True, pin_memory = arg.pm, num_workers=args.wk)

    return args, train_dl, val_dl, test_dl