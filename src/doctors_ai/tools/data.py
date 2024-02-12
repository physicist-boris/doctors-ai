import subprocess
from pathlib import Path
from typing import Tuple
"""This module includes the functions to prepare datasets for ML applications.
"""
import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split



@dataclass
class Dataset:
    """A dataclass used to represent a Dataset

    Attributes
    ----------
    train_x : Pandas DataFrame
        the dataframe of input features w.r.t training split
    val_x : Pandas DataFrame
        the dataframe of input features w.r.t validation split
    test_x : Pandas DataFrame
        the dataframe of input features w.r.t testing split
    train_y : Pandas Series
        the series of output label w.r.t training split
    val_y : Pandas Series
        tthe series of output label w.r.t validation split
    test_y : Pandas Series
        the series of output label w.r.t testing split
    """

    train_x: pd.DataFrame
    val_x: pd.DataFrame
    test_x: pd.DataFrame
    train_y: pd.Series
    val_y: pd.Series
    test_y: pd.Series

    def merge_in(self, dataset):
        self.train_x = pd.concat([self.train_x, dataset.train_x], axis=0)
        self.val_x = pd.concat([self.val_x, dataset.val_x], axis=0)
        self.test_x = pd.concat([self.test_x, dataset.test_x], axis=0)
        self.train_y = pd.concat([self.train_y, dataset.train_y], axis=0)
        self.val_y = pd.concat([self.val_y, dataset.val_y], axis=0)
        self.test_y = pd.concat([self.test_y, dataset.test_y], axis=0)

    def persist(self, dirpath):
        self.train_x.to_csv(os.path.join(dirpath,'train_x.csv'), sep=';', index=False)
        self.train_y.to_csv(os.path.join(dirpath, 'train_y.csv'), sep=';', index=False)
        self.val_x.to_csv(os.path.join(dirpath, 'val_x.csv'), sep=';', index=False)
        self.val_y.to_csv(os.path.join(dirpath, 'val_y.csv'), sep=';', index=False)
        self.test_x.to_csv(os.path.join(dirpath,'test_x.csv'), sep=';', index=False)
        self.test_y.to_csv(os.path.join(dirpath, 'test_y.csv'), sep=';', index=False)

def autodetect_commit() -> str:
    """Retrieves the current commit of the local repository

    Assumes the repository is local.
    Reads the HEAD value.

    Returns:
        The commit SHA
    """
    output = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--no-flags",
            "--tags",
            "HEAD",
        ],
        text=True,
    )
    return output.strip()



def load_dataset_from_localfs(
    data_dir: str = "data", with_dvc_info: bool = True, with_vcs_info: bool = True
) -> Tuple[Dataset, dict]:
    """Loads a Dataset object coming from DVC


    As of this implementation, reads the CSV files that are locally stored

    Args:
        data_dir(str): Directory in which data resides. Defaults to "data/".
    Returns:
        A Tuple containing:
            in the first position, the Dataset object
            in the second position, the metadata information
    """
    data_dir = Path(data_dir)

    dataset_info = {}

    d = Dataset(
        train_x=pd.read_csv(f"{data_dir}/splits/train_x.csv", sep=';'),
        val_x=pd.read_csv(f"{data_dir}/splits/val_x.csv", sep=';'),
        test_x=pd.read_csv(f"{data_dir}/splits/test_x.csv", sep=';'),
        train_y=pd.read_csv(f"{data_dir}/splits/train_y.csv", sep=';').iloc[:, 0],
        val_y=pd.read_csv(f"{data_dir}/splits/val_y.csv", sep=';').iloc[:, 0],
        test_y=pd.read_csv(f"{data_dir}/splits/test_y.csv", sep=';').iloc[:, 0]
    )

    return d, dataset_info


def load_prep_dataset_from_minio(
    minio_client,
    data_bucket: str = "training-data-bucket",
) -> Tuple[Dataset, dict]:
    
    obj = minio_client.get_object(data_bucket, "training_dataset.csv")
    df = pd.read_csv(obj)
    X= df.drop(columns=["admitted"])
    y = df["admitted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    d = Dataset(
        train_x=X_train,
        val_x=X_val,
        test_x=X_test,
        train_y=y_train,
        val_y=y_val,
        test_y=y_test
    )
    
    return d


def load_raw_datasets_from_minio(
    minio_client,
    data_bucket: str = "training-data-bucket"):
    ds_info = {}
    obj = minio_client.get_object(data_bucket, "training_dataset.csv")
    df = pd.read_csv(obj)
    ds_info["training_dataset"] = len(df)
    return df, ds_info

        

def prepare_binary_classfication_tabular_data(
    data_frame: pd.DataFrame,
    predictors: List[str],
    predicted: str,
    pos_neg_pair: Tuple[str, str] | None = None,
    splits_sizes: Tuple[float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Dataset:
    """Prepare the training/validation/test inputs (X) and outputs (y) for binary clasification modeling

    Args:
    ----
        data_frame (pd.DataFrame): aggregated data frame
        predictors (List[str]): list of predictors column names
        predicted (str): column name of the predicted outcome
        pos_neg_pair (Tuple[str,str], optional): groundtruth positive/negative labels. Defaults to None.
        splits_sizes (List[float], optional): list of relative size portions for training, validation, test data, respectively. Defaults to [0.7,0.1,0.2].
        seed (int, optional): random seed. Defaults to 42.

    Returns:
    -------
        Dataset: datassets for binary classification with training/validation/test splits
    """
    X = data_frame[predictors].copy()  # noqa
    y = data_frame[predicted].copy()
    if pos_neg_pair:
        postive, negative = pos_neg_pair
        y = y.replace({postive: 1, negative: 0})
    train_size, valid_size, test_size = splits_sizes
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=seed)
    valid_size /= train_size + valid_size
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=valid_size, random_state=seed)
    dataset = Dataset(train_x, val_x, test_x, train_y, val_y, test_y)
    return dataset


def prepare_binary_classfication_tabular_data_from_splits(
    csv_dirpath: str,
    predictors: List[str],
    predicted: str,
    pos_neg_pair: Tuple[str, str] | None = None,
    splits_sizes: Tuple[float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Dataset:
    """Prepare the training/validation/test inputs (X) and outputs (y) for binary clasification modeling

    Args:
    ----
        csv_dirpath (str): path of the directory of csv files
        predictors (List[str]): list of predictors column names
        predicted (str): column name of the predicted outcome
        pos_neg_pair (Tuple[str,str], optional): groundtruth positive/negative labels. Defaults to None.
        splits_sizes (List[float], optional): list of relative size portions for training, validation, test data, respectively. Defaults to [0.7,0.1,0.2].
        seed (int, optional): random seed. Defaults to 42.

    Returns:
    -------
        Dataset: datassets for binary classification with training/validation/test splits
    """
    dataset = None
    for fname in os.listdir(csv_dirpath):
        if not fname.endswith('.csv'): continue
        fpath = os.path.join(csv_dirpath, fname)
        data_frame = pd.read_csv(fpath)
        if dataset != None:
            dataset.merge_in(prepare_binary_classfication_tabular_data(data_frame, predictors, predicted, 
                                                                       pos_neg_pair, splits_sizes, seed))
        else:
            dataset = prepare_binary_classfication_tabular_data(data_frame, predictors, predicted, 
                                                                pos_neg_pair, splits_sizes, seed)
    return dataset


def prepare_and_merge_splits_to_dataset(
    dataset,
    dataframes,
    predictors: List[str],
    predicted: str,
    pos_neg_pair: Tuple[str, str] | None = None,
    splits_sizes: Tuple[float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Dataset:
    for data_frame in dataframes:
        dataset.merge_in(prepare_binary_classfication_tabular_data(data_frame, predictors, predicted, 
                                                                    pos_neg_pair, splits_sizes, seed))
    return dataset