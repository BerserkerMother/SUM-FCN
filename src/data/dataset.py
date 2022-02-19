import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import os
import numpy as np
import h5py

from .path import PATH


# Dataset Implementation for DS-net TVsum & SumMe

class TSDataset(Dataset):
    def __init__(self, root, dataset, key):
        self.root = root
        self.key = key
        self.files_name = self.get_datasets(self.key)

        self.data = []
        self.target = []
        self.user_summaries = []
        with h5py.File(os.path.join(root, PATH[dataset]), 'r') as f:
            for key in self.files_name:
                self.data.append(f[key]['features'][...].astype(np.float32))
                self.target.append(f[key]['gtscore'][...].astype(np.float32))
                user_summary = np.array(f[key]['user_summary'])
                user_scores = np.array(f[key]["user_scores"])
                sb = np.array(f[key]['change_points'])
                n_frames = np.array(f[key]['n_frames'])
                positions = np.array(f[key]['picks'])

                self.user_summaries.append(
                    UserSummaries(user_summary, user_scores, key,
                                  sb, n_frames, positions))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx])
        targets = torch.tensor(self.target[idx])

        # randomly sample 320 frames of videos
        num_frames = features.size()[0]
        random_indices = np.random.choice(num_frames, size=320)
        features = features[random_indices, :]
        targets = targets[random_indices]
        sampling = (num_frames, random_indices)
        return features, targets, sampling, self.user_summaries[idx]

    def get_datasets(self, keys: List[str]):
        files_name = [str(Path(key).name) for key in keys]
        # datasets = [h5py.File(path, 'r') for path in dataset_paths]
        return files_name


class PreTrainDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.data = []
        self.target = []
        self.name = []
        for dataset in PATH.keys():
            with h5py.File(os.path.join(root, PATH[dataset]), 'r') as f:
                for key in f.keys():
                    self.data.append(f[key]['features']
                                     [...].astype(np.float32))
                    self.name.append(key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx])
        vid_name = self.name[idx]

        return features, vid_name


class UserSummaries:
    def __init__(self, user_summary, user_scores, name,
                 changes_point, n_frames, picks):
        self.user_summary = user_summary
        self.user_scores = user_scores
        self.change_points = changes_point
        self.n_frames = n_frames
        self.picks = picks
        self.name = name


def collate_fn(batch):
    features, targets, samplings, user_summaries = batch[0]
    features = features.unsqueeze(0)
    targets = targets.unsqueeze(0)
    return features, targets, samplings, user_summaries
