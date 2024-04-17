import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import json
import os
from pyntcloud import PyntCloud
import point_cloud_utils as pcu
import pandas as pd
from typing import Optional
from tqdm import tqdm
from multiprocessing import Pool
import pyshtools
from scipy import interpolate as spinterp
import torch.nn.functional as F
from sklearn import preprocessing



class CellPackDataModule2(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_points: int,
        num_points_ref: int,
        structure_path='/allen/aics/modeling/ritvik/forSaurabh/npm1/',
        ref_path="/allen/aics/modeling/ritvik/forSaurabh/",
        return_id=False,
        x_label=None,
        ref_label=None,
        scale: Optional[float] = 1,
        rotation_augmentations: Optional[int] = None,
        jitter_augmentations: Optional[int] = None,
        max_ids: Optional[int] = None,
        norm_feats: Optional[bool] = False,
        num_rotations: Optional[int] = 10,
    ):
        """ """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader_fnc = DataLoader
        self.num_points = num_points
        self.return_id = return_id
        self.scale = scale
        self.rotation_augmentations = rotation_augmentations
        self.jitter_augmentations = jitter_augmentations
        self.num_points_ref = num_points_ref
        self.x_label = x_label
        self.ref_label = ref_label
        self.structure_path = structure_path
        self.ref_path = ref_path
        self.max_ids = max_ids
        self.norm_feats = norm_feats
        self.num_rotations = num_rotations

    def _get_dataset(self, split):
        return CellPackDataset(
            self.num_points,
            self.num_points_ref,
            self.return_id,
            self.structure_path,
            self.ref_path,
            split,
            self.x_label,
            self.ref_label,
            self.scale,
            self.max_ids,
            self.norm_feats,
            self.num_rotations
        )

    def train_dataloader(self):
        self.train_dataset = self._get_dataset("train")
        dataloader = self.loader_fnc(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        self.val_dataset = self._get_dataset("valid")
        dataloader = self.loader_fnc(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        self.test_dataset = self._get_dataset("test")
        dataloader = self.loader_fnc(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader


class CellPackDataset(Dataset):
    def __init__(
        self,
        num_points,
        num_points_ref,
        return_id=False,
        structure_path='/allen/aics/modeling/ritvik/forSaurabh/npm1/',
        ref_path="/allen/aics/modeling/ritvik/forSaurabh/",
        split="train",
        x_label: str = "pcloud",
        ref_label: str = "nuc",
        scale: Optional[float] = 1,
        max_ids: Optional[int] = None,
        norm_feats: Optional[bool] = False,
        num_rotations: Optional[int] = 3,
    ):
        self.x_label = x_label
        self.scale = scale
        self.ref_label = ref_label
        self.num_points = num_points
        self.structure_path = structure_path
        self.ref_path = ref_path
        self.num_points_ref = num_points_ref
        self.max_ids = max_ids
        self.num_rotations = num_rotations

        self.ref_csv = pd.read_csv(ref_path + "manifest.csv")

        self.return_id = return_id
        self.split = split
        items = os.listdir(self.structure_path)

        self._all_ids = list([i.split('_')[-3] for i in items])
        self._all_ids = np.unique(np.array(self._all_ids))
        self._all_ids = list(self._all_ids)
        if self.max_ids:
            self._all_ids = self._all_ids[: self.max_ids]

        _splits = {
            "train": self._all_ids[: int(0.7 * len(self._all_ids))],
            "valid": self._all_ids[
                int(0.7 * len(self._all_ids)) : int(0.85 * len(self._all_ids))
            ],
            "test": self._all_ids[int(0.85 * len(self._all_ids)) :],
        }

        self.ids = _splits[split]

        df_orig = pd.read_csv('/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv')

        if norm_feats:
            x = self.ref_csv[['shape_volume_lcc', 'roundness_surface_area', 'position_depth']].values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            self.ref_csv[['shape_volume_lcc', 'roundness_surface_area', 'position_depth']] = pd.DataFrame(x_scaled)

        for feat in ['shape_volume_lcc', 'roundness_surface_area', 'position_depth']:
            self.ref_csv[f'{feat}_bins'] = pd.cut(self.ref_csv[feat], bins=5)
            self.ref_csv[f'{feat}_bins'] = pd.factorize(self.ref_csv[f'{feat}_bins'])[0]


        tup = []
        self.num_rules = 3
        # self.num_rotations = 10

        for this_id in tqdm(self.ids, total=len(self.ids)):
            for clusters in [1, 2, 3]:
                for i in range(self.num_rotations):
                    rotation_bool = True
                    if i == 0:
                        rotation_bool = False
                    this_path = (
                        "voxelized_image_npm1_analyze_"
                        + f"{clusters}_clust_random_"
                        + f"{this_id}"
                        + f"_seed_0"
                        + ".ply"
                    )
                    if os.path.isfile(structure_path + this_path):
                        nuc_path = ref_path + f"{this_id}_0.obj"
                        nuc_vol2 = df_orig.loc[df_orig['CellId'] == this_id]['volume_of_nucleus_um3'].item()
                        nuc_height = self.ref_csv.loc[self.ref_csv['CellId'] == this_id]['position_depth'].iloc[0]
                        nuc_area = self.ref_csv.loc[self.ref_csv['CellId'] == this_id]['roundness_surface_area'].iloc[0]
                        nuc_vol = self.ref_csv.loc[self.ref_csv['CellId'] == this_id]['shape_volume_lcc'].iloc[0]
                        nuc_vol_bin = self.ref_csv.loc[self.ref_csv['CellId'] == this_id]['shape_volume_lcc_bins'].iloc[0]

                        tup.append(
                            [
                                structure_path + this_path,
                                self._all_ids.index(this_id),
                                clusters,
                                nuc_path,
                                num_points_ref,
                                num_points,
                                nuc_vol,
                                nuc_height,
                                nuc_area,
                                nuc_vol2,
                                nuc_vol_bin,
                                rotation_bool,
                                False
                            ]
                        )

        # get_packing(tup[0])
        with Pool(3) as p:
            all_packings = tuple(
                tqdm(
                    p.imap_unordered(
                        get_packing,
                        tup,
                    ),
                    total=len(tup),
                    desc="get_packings",
                )
            )

        self.data = [i[0] for i in all_packings]
        self.ref = [i[1] for i in all_packings]
        self.id_list = [i[2] for i in all_packings]
        self.rule = [i[3] for i in all_packings]
        self.rot = [i[4] for i in all_packings]
        self.jitter = [i[5] for i in all_packings]
        self.nuc_vol = [i[6] for i in all_packings]
        self.nuc_height = [i[7] for i in all_packings]
        self.nuc_area = [i[8] for i in all_packings]
        self.nuc_vol2 = [i[9] for i in all_packings]
        self.nuc_vol_bins = [i[10] for i in all_packings]

        self.len = len(self.data)
        self.label = []

    def __len__(self):
        return self.len

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        x = self.data[item]
        # x = self.pc_norm(x)
        x = torch.from_numpy(x).float() * self.scale
        ref = self.ref[item] * self.scale
        # ref = self.pc_norm(ref)
        ref = torch.from_numpy(ref).float()
        if self.return_id:
            return {
                self.x_label: x,
                self.ref_label: ref,
                "CellId": torch.tensor(self.id_list[item]).unsqueeze(dim=0),
                "rule": torch.tensor(self.rule[item]).unsqueeze(dim=0),
                "rotation_aug": torch.tensor(self.rot[item]),
                "jitter_aug": torch.tensor(self.jitter[item][0]).unsqueeze(dim=0),
                "nuc_vol": torch.tensor(self.nuc_vol[item]).unsqueeze(dim=0).float(),
                "nuc_height": torch.tensor(self.nuc_height[item]).unsqueeze(dim=0).float(),
                "nuc_area": torch.tensor(self.nuc_area[item]).unsqueeze(dim=0).float(),
                "nuc_vol2": torch.tensor(self.nuc_vol2[item]).unsqueeze(dim=0).float(),
                "nuc_vol_bin": torch.tensor(self.nuc_vol_bins[item]).unsqueeze(dim=0).long(),
                "rule_one_hot": F.one_hot(torch.tensor(self.rule[item]).unsqueeze(dim=0), self.num_rules).squeeze(),
                "structure": torch.tensor([1]).unsqueeze(dim=0).float(),
            }
        else:
            return {self.x_label: x, self.ref_label: ref}


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud, rotation_matrix=None, return_rot=False):
    pointcloud_rotated = pointcloud.copy()
    if rotation_matrix is None:
        theta = np.pi * 2 * np.random.choice(24) / 24
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    pointcloud_rotated[:, [0, 1]] = pointcloud_rotated[:, [0, 1]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    if return_rot:
        return pointcloud_rotated, rotation_matrix, theta

    return pointcloud_rotated


def _subsample(points_dna, num_points_ref):
    v = points_dna.values
    idx = pcu.downsample_point_cloud_poisson_disk(v, num_samples=num_points_ref + 50)
    v_sampled = v[idx]
    points_dna = pd.DataFrame()
    points_dna["x"] = v_sampled[:, 0]
    points_dna["y"] = v_sampled[:, 1]
    points_dna["z"] = v_sampled[:, 2]
    points_dna = points_dna.sample(n=num_points_ref)
    return points_dna


def get_packing(tup):
    this_path = tup[0]
    this_index = tup[1]
    this_rule = tup[2]
    nuc_path = tup[3]
    num_points_ref = tup[4]
    num_points = tup[5]
    nuc_vol = tup[6]
    nuc_height = tup[7]
    nuc_area = tup[8]
    nuc_vol2 = tup[9]
    nuc_vol_bins = tup[10]
    rotate = tup[11]
    jitter = tup[12]

    pc = PyntCloud.from_file(this_path)
    points = pc.points

    my_point_cloud = PyntCloud.from_file(nuc_path)
    points_dna = my_point_cloud.points

    nuc_she = get_shcoeffs(points_dna.values).values.astype(np.float32).flatten()

    if points.shape[0] > num_points:
        points = _subsample(points, num_points)

    if rotate:
        (
            points_dna,
            rotation_matrix,
            theta,
        ) = rotate_pointcloud(points_dna.values, return_rot=True)
        points = rotate_pointcloud(points.values, rotation_matrix, return_rot=False)
        theta = np.array([theta])
    else:
        theta = np.array([0])
        points = points.values
        points_dna = points_dna.values

    if jitter:
        points = jitter_pointcloud(points)
        points_dna = jitter_pointcloud(points_dna)
        jitter_ret = np.array([1])
    else:
        jitter_ret = np.array([0])

    return (
        points,
        nuc_she,
        this_index,
        this_rule - 1,
        theta,
        jitter_ret,
        nuc_vol,
        nuc_height, 
        nuc_area,
        nuc_vol2,
        nuc_vol_bins,
    )


def get_shcoeffs(points, lmax=16):
    # import ipdb

    # ipdb.set_trace()
    x = points[:, 2]
    y = points[:, 1]
    z = points[:, 0]
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # # Reconstruct grid. Look at pyshtools for detail.
    # grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)
    keys = []
    for suffix in ["C", "S"]:
        for L, m in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{L}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))
    coeffs_dict = dict((f"{k}_lcc", v) for k, v in coeffs_dict.items())

    coeffs_dict = pd.DataFrame(coeffs_dict, index=[0])
    coeffs_dict = coeffs_dict.loc[:, (coeffs_dict != 0).any(axis=0)]
    return coeffs_dict
