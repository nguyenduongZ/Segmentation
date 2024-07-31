import cv2
import torch
import torchvision
import torchvision.datasets
import albumentations as A
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm

_root = "/media/mountHDD3/data_storage"

def get_trans_lst():
    return [
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 10.0)),
            A.MultiplicativeNoise(),
            A.RandomRain(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15, p=0.5,
                          border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),   
            A.Downscale(interpolation = {
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_NEAREST
            }),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.3),
            A.ColorJitter(p=0.3),
        ], p= 0.3),
        A.RGBShift(p=0.3),
        A.RandomShadow(p=0.2)
    ]

def get_trans_nonnoise_nonblur_lst():
    return [
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15, p=0.5,
                          border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),   
            A.Downscale(interpolation = {
                "downscale": cv2.INTER_NEAREST,
                "upscale": cv2.INTER_NEAREST
            }),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(p=0.3),
            A.ColorJitter(p=0.3),
        ], p= 0.3),
        A.RGBShift(p=0.3)
    ]

class OxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(
            self,
            root:str,
            split:str,
            target_types = "segmetation",
            download = False,
            transform = None,
            target_transform = None,
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
                A.Resize(512, 512),
            ]
        )

        # self.aug_transforms = A.Compose(
        #     [
        #         A.HorizontalFlip(p=0.5),
        #         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
        #     ]
        # )
        norm = True 
        self.aug_transforms = A.Compose(get_trans_lst() if norm else get_trans_nonnoise_nonblur_lst(), p = 0.9)

        self.norm = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self._images_folder = self.root + "/oxford-iiit-pet/images"
        self._anns_folder = self.root + "/oxford-iiit-pet/annotations"
        self._segs_folder = self._anns_folder + "/trimaps"

        print("Data Set Setting Up")
        image_ids = []
        self._labels = []
        with open(self._anns_folder + f"/{self._split}.txt") as file:
            for line in tqdm(file):
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in tqdm(
                    sorted(
                    {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                    key=lambda image_id_and_label: image_id_and_label[1],
                )
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder + f"/{image_id}.jpg" for image_id in tqdm(image_ids)]
        self._segs = [self._segs_folder + f"/{image_id}.png" for image_id in tqdm(image_ids)]
        print("Done")

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

        target = {
            "semantic" : self.process_mask(torch_mask),
            "category" : self._labels[idx],
        }

        return torch_img, target

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in ['train', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'test']")
        else:
            self.__mode = m

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