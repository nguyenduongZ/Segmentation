import cv2
import torch
import torchvision
import torchvision.datasets
import albumentations as A
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, random_split
from PIL import Image

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
        self._split = split
        self.__mode = "train" if self._split == 'trainval' else 'test'

        self.resize = A.Compose(
            [
                A.Resize(256, 256),
            ]
        )

        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )

        self.norm = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def process_mask(x):
        uniques = torch.unique(x, sorted = True)
        if uniques.shape[0] > 3:
            x[x == 0] = uniques[2]
            uniques = torch.unique(x, sorted = True)
        for i, v in enumerate(uniques):
            x[x == v] = i
        
        x = x.to(dtype=torch.long)
        onehot = F.one_hot(x.squeeze(1), 3).permute(0, 3, 1, 2)[0].float()
        return onehot

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        (image, mask) = super().__getitem__(idx)

        image = np.array(Image.open(self._images[idx]).convert("RGB"))
        mask = np.array(Image.open(self._segs[idx])) 

        resized = self.resize(image=image, mask=mask)

        if self.__mode == 'train':
            transformed = self.aug_transforms(image=resized['image'], mask=resized['mask'])
            transformed_img = self.norm(image=transformed['image'])['image']
            transformed_msk = transformed['mask']
            
        else:
            transformed_img = self.norm(image=resized['image'])['image']
            transformed_msk = resized['mask']
            
        torch_img = torch.from_numpy(transformed_img).permute(-1, 0, 1).float()
        torch_mask = torch.from_numpy(transformed_msk).unsqueeze(-1).permute(-1, 0, 1).float()

        # print(torch_mask.shape())

        return torch_img, self.process_mask(torch_mask)

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in ['train', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'test']")
        else:
            self.__mode = m
# transform = A.Compose(
#     [
#         A.Resize(256, 256),
#         A.HorizontalFlip(p=0.2),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ]
# )

# target_transform = A.Compose(
#     [
#         A.Resize(256, 256),
#         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ]
# )

def get_ds(args):
    train_data = OxfordIIITPet(
        root = _root,
        split = "trainval",
        target_types = "segmentation",
        download = False,
        # transform = transform
    )

    test_data = OxfordIIITPet(
        root = _root,
        split = "test",
        target_types = "segmentation",
        download = False,
        # target_transform = target_transform
    )

    val_size = int(len(train_data) * 0.1)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_dl = DataLoader(train_data, batch_size=args.bs, shuffle=True, pin_memory = args.pm, num_workers=args.wk)

    val_dl = DataLoader(val_data, batch_size=args.bs, shuffle=True, pin_memory = args.pm, num_workers=args.wk)

    test_dl = DataLoader(test_data, batch_size=args.bs, shuffle=True, pin_memory = args.pm, num_workers=args.wk)

    return args, train_dl, val_dl, test_dl