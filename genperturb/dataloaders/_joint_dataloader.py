import pandas as pd
import polars as pol

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from genperturb.dataloaders._genome import GenomeIntervalDataset

import h5py
from typing import Optional

class HDF5Dataset:
    def __init__(self, hdf5, indices=None):
        self.hdf5 = hdf5
        self.file = h5py.File(self.hdf5, 'r')
        self.fasta_dataset_indices = indices

    def __getitem__(self, idx):
        actual_idx = self.fasta_dataset_indices[idx] if self.fasta_dataset_indices else idx
        return self.file['embedding'][actual_idx].flatten()

    def __del__(self):
        if self.file:
            self.file.close()

class JointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df : pd.DataFrame = None,
        hdf5 : Optional[str] = None,
        bed : Optional[str] = None,
        fasta : Optional[str] = None,
        return_sequence : Optional[bool] = False,
        datasplit : Optional[str] = None,
        context_length : int = 196_608,
        tokenizer = None,
    ):

        self.hdf5 = hdf5
        self.bed = bed
        self.fasta = fasta
        self.return_sequence = return_sequence
        self.tokenizer = tokenizer

        if self.hdf5 is not None:
            indices = df.loc[:, "training"].reset_index().query('training == @datasplit').index.to_list()
            self.hdf5_dataset = HDF5Dataset(hdf5, indices)
            self.target = torch.tensor(df.query('training == @datasplit').drop(labels="training", axis=1).astype("float32").values)
        elif self.bed is not None and self.fasta is not None:
            self.fasta_dataset = self.create_seq(bed, fasta, context_length)
            self.target = torch.tensor(df.query('training == @datasplit').drop(labels="training", axis=1).astype("float32").values)

    def create_seq(self, bed, fasta, context_length):
        ds = GenomeIntervalDataset(
            bed_file = bed,
            fasta_file = fasta,
            #return_seq_indices = True,
            return_sequence = self.return_sequence,
            #shift_augs = (-2, 2),
            #rc_aug = True,
            context_length = context_length
        )
        return ds

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if self.hdf5 is not None:
            return [self.hdf5_dataset[idx], self.target[idx]]

        elif self.bed is not None and self.fasta is not None:
            seq = self.fasta_dataset[idx]
        
            if self.tokenizer is not None:
                seq = torch.LongTensor(self.tokenizer(seq, truncation=True)["input_ids"])

            return [seq, self.target[idx]]


class JointDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        df : pd.DataFrame = None,
        hdf5 : str = None,
        bed : Optional[str] = None,
        fasta : Optional[str] = None,
        return_sequence : Optional[bool] = False,
        context_length : int = 196_608,
        batch_size : int = 1,
        num_workers : int = 1,
        tokenizer = None,
    ):
        super().__init__()
        self.df = df
        self.hdf5 = hdf5
        self.bed = bed
        self.fasta = fasta
        self.return_sequence = return_sequence
        self.batch_size = batch_size
        self.num_workers = num_workers


        if self.hdf5 is not None:
            self.dfs = [
                JointDataset(
                    df=self.df,
                    hdf5=self.hdf5,
                    datasplit=i,
                    context_length=context_length
                )
                for i in ["train", "val", "test"]
            ]
        elif self.bed is not None and self.fasta is not None:
            self.dfs = [
                JointDataset(
                    df=self.df,
                    bed=self.bed,
                    fasta=self.fasta,
                    datasplit=i,
                    return_sequence=self.return_sequence,
                    context_length=context_length,
                    tokenizer=tokenizer
                )
                for i in ["train", "val", "test"]
            ]

    def train_dataloader(self):
        return DataLoader(self.dfs[0], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dfs[1], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dfs[2], batch_size=self.batch_size, num_workers=self.num_workers)


class JointDatasetPred(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5: str = None,
        bed: Optional[str] = None,
        fasta: Optional[str] = None,
        return_sequence : Optional[bool] = False,
        context_length: int = 196_608,
        tokenizer = None,
        mask = None,
    ):

        self.hdf5 = hdf5
        self.bed = bed
        self.fasta = fasta
        self.return_sequence = return_sequence
        self.tokenizer = tokenizer
        self.mask = mask
    

        if self.hdf5 is not None:
            self.hdf5_dataset = HDF5Dataset(hdf5)
        elif self.bed is not None and self.fasta is not None:
            self.fasta_dataset = self.create_seq(bed, fasta, context_length)

    def create_seq(self, bed, fasta, context_length=196_608):
        ds = GenomeIntervalDataset(
            bed_file=bed,
            fasta_file=fasta,
            return_sequence = self.return_sequence,
            context_length=context_length
        )
        return ds

    def __len__(self):
        return len(self.fasta_dataset) if self.bed is not None else self.hdf5_dataset.file['embedding'].shape[0]

    def __getitem__(self, idx):
        if self.hdf5 is not None:
            return [self.hdf5_dataset[idx]]
        elif self.bed is not None and self.fasta is not None:
            seq = self.fasta_dataset[idx]

            if self.tokenizer is not None:
                seq = torch.LongTensor(self.tokenizer(seq, truncation=True)["input_ids"])

            if self.mask is not None:
                seq[:self.mask[0],:] = 0
                seq[self.mask[1]:,:] = 0
            
            return [seq]



class JointDataLoaderPred(pl.LightningDataModule):
    def __init__(
        self,
        hdf5: str = None,
        bed: Optional[str] = None,
        fasta: Optional[str] = None,
        return_sequence : Optional[bool] = False,
        context_length: int = 196_608,
        mask = None,
        batch_size: int = 64,
        num_workers: int = 1,
        tokenizer = None,
        sequence=False,
    ):
        super().__init__()
        self.hdf5 = hdf5
        self.bed = bed
        self.fasta = fasta
        self.return_sequence = return_sequence
        self.context_length = context_length
        self.mask = mask
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.hdf5 is not None:
            self.loader = JointDatasetPred(hdf5=self.hdf5)
        elif self.bed is not None and self.fasta is not None:
            self.loader = JointDatasetPred(
                bed=self.bed,
                fasta=self.fasta,
                context_length=self.context_length,
                tokenizer=tokenizer,
                return_sequence=self.return_sequence,
                mask=self.mask
            )

    def dataloader(self):
        return DataLoader(self.loader, batch_size=self.batch_size, num_workers=self.num_workers)
