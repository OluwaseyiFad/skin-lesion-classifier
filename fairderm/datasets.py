# dataset classes for fitzpatrick17k and DDI
# both return (image, label, skin_tone_group) tuples

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    # fitzpatrick17k dataset - main training dataset
    # heavily imbalanced in skin tones (mostly light skin)
    # expects df with columns: image_path, label_num (0/1), tone_group (Light/Medium/Dark/Unknown)

    def __init__(self, df, transform=None, return_group=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_group = return_group

        # map string group names to integers
        self.group_to_idx = {
            'Light': 0,
            'Medium': 1,
            'Dark': 2,
            'Unknown': 3
        }
        self.idx_to_group = {v: k for k, v in self.group_to_idx.items()}

        # convert group names to indices, handle any weird values
        mapped = self.df['tone_group'].map(self.group_to_idx)
        n_unmapped = mapped.isna().sum()
        if n_unmapped > 0:
            unmapped_values = self.df.loc[mapped.isna(), 'tone_group'].unique().tolist()
            print(f"Warning: {n_unmapped} samples have unmapped tone_group values "
                  f"{unmapped_values}, treating as 'Unknown'")
        self.group_indices = mapped.fillna(self.group_to_idx['Unknown']).astype(int).values

        # precompute which samples belong to each group
        # the adaptive sampler needs this to sample by group
        self.group_sample_indices = {}
        for group, idx in self.group_to_idx.items():
            mask = self.group_indices == idx
            self.group_sample_indices[group] = np.where(mask)[0].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label_num'], dtype=torch.long)

        if self.return_group:
            group_idx = torch.tensor(self.group_indices[idx], dtype=torch.long)
            return image, label, group_idx

        return image, label

    def get_group_indices(self, group_name):
        # returns list of sample indices that belong to this group
        # used by adaptive sampler to weight groups differently
        return self.group_sample_indices.get(group_name, [])


class DDIDataset(Dataset):
    # DDI dataset - used for evaluation only
    # intentionally balanced across skin tones, so it's a fairer test set
    # skin_tone values: 12=Light, 34=Medium, 56=Dark

    SKIN_TONE_MAP = {
        12: 'Light',
        34: 'Medium',
        56: 'Dark'
    }

    GROUP_TO_IDX = {
        'Light': 0,
        'Medium': 1,
        'Dark': 2
    }

    def __init__(self, metadata_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        self.df = pd.read_csv(metadata_path)

        # map DDI's numeric skin tone codes to our group names
        self.df['skin_tone_group'] = self.df['skin_tone'].map(self.SKIN_TONE_MAP)
        unknown_mask = self.df['skin_tone_group'].isna()
        if unknown_mask.any():
            missing_codes = self.df.loc[unknown_mask, 'skin_tone'].unique().tolist()
            print(
                f"Warning: {unknown_mask.sum()} entries have unknown skin_tone codes {missing_codes}; dropping."
            )
            self.df = self.df[~unknown_mask].reset_index(drop=True)

        # malignant column: 0=benign, 1=malignant
        self.df['label'] = self.df['malignant'].astype(int)

        self._verify_images()

        # print summary
        print(f"\nDDI Dataset loaded: {len(self.df)} images")
        print(f"\nSkin tone distribution:")
        print(self.df['skin_tone_group'].value_counts())
        print(f"\nClass distribution:")
        print(self.df['label'].value_counts().rename({0: 'Benign', 1: 'Malignant'}))

    def _verify_images(self):
        # make sure all image files actually exist
        missing = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.images_dir, row['DDI_file'])
            if not os.path.exists(img_path):
                missing.append(row['DDI_file'])

        if missing:
            print(f"Warning: {len(missing)} images not found, removing from dataset")
            self.df = self.df[~self.df['DDI_file'].isin(missing)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.images_dir, row['DDI_file'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label'], dtype=torch.long)
        group_name = row['skin_tone_group']
        if group_name not in self.GROUP_TO_IDX:
            raise KeyError(f"Unmapped skin_tone_group encountered: {group_name}")
        group_idx = torch.tensor(self.GROUP_TO_IDX[group_name], dtype=torch.long)

        return image, label, group_idx

    def get_group_indices(self, group_name):
        # returns list of sample indices for a skin tone group
        mask = self.df['skin_tone_group'] == group_name
        return self.df[mask].index.tolist()

    def get_statistics(self):
        # summary stats for the dataset
        return {
            'total': len(self.df),
            'per_group': self.df['skin_tone_group'].value_counts().to_dict(),
            'per_class': self.df['label'].value_counts().to_dict(),
            'per_group_per_class': self.df.groupby(
                ['skin_tone_group', 'label']
            ).size().to_dict()
        }
