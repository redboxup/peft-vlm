import random
import pandas as pd
from torch.utils.data import Sampler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from utils.datasets import CXRDatasets

class NShotMultiClassSampler(Sampler):
    """PyTorch Sampler for creating N-shot MultiClass datasets for 
    few-shot prompt learning based methods.

    If not enough single-label samples exist for a class, fill the rest
    using multi-label samples.
    """
    def __init__(
        self,
        dataset: CXRDatasets,
        n_shot: int = 2,
        random_seed: int = 42
    ):
        self.dataset = dataset
        self.n_shot = n_shot
        self.random_seed = random_seed
        self.num_classes = len(dataset.cxr_labels)

        self.df: pd.DataFrame = dataset.df[
            list(dataset.cxr_labels)
        ].copy(deep=True)

        self.class_indices = self._get_single_label_indices()
        # print(f"Multiclass Sampler initialized with {self.n_shot} shots")

    def _get_single_label_indices(self):
        """Get indices of samples with single labels."""
        classwise_indices = []
        all_indices = []

        pathology_cols = list(self.dataset.cxr_labels)
        self.df['pathology_count'] = self.df[pathology_cols].sum(axis=1)

        df_single = self.df[self.df['pathology_count'] == 1].copy()
        df_multi = self.df[self.df['pathology_count'] > 1].copy()

        for pathology in pathology_cols:
            single_positive = df_single[df_single[pathology] == 1]

            if len(single_positive) >= self.n_shot:
                sampled = single_positive.sample(
                    n=self.n_shot,
                    replace=False,
                    random_state=self.random_seed
                )

            else:
                sampled = single_positive.copy()
                deficit = self.n_shot - len(sampled)

                multi_positive = df_multi[df_multi[pathology] == 1]
                multi_sample = multi_positive.sample(
                    n=deficit, replace=True, random_state=self.random_seed
                )
                sampled = pd.concat([sampled, multi_sample])

            indices = sampled.index.to_list()
            classwise_indices.append(indices)
            all_indices.extend(indices)

        self.df.drop(columns=['pathology_count'], inplace=True)
        return all_indices

    def __iter__(self):
        """
        Returns an iterator that yields indices for n-shot sampling.
        """
        indices = self.class_indices
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        """Returns the total number of samples in the sampler."""
        return self.n_shot * self.num_classes

class NShotRandomSampler(Sampler):
    def __init__(
            self,
            dataset: CXRDatasets,
            n_shot: int = 2,
            random_seed: int = 42
        ):
        self.dataset = dataset
        self.n_shot = n_shot
        self.num_classes = len(dataset.cxr_labels)
        self.random_seed = random_seed

        self.df = dataset.df[
            list(dataset.cxr_labels)
        ].copy(deep=True)
        self.class_indices = self._get_random_indices()
        # print(f"Random Sampler initialized with {self.n_shot} shots")

    def _get_random_indices(self):
        """ Get random indices for n-shot sampling.
        """
        num_samples = self.n_shot * self.num_classes

        all_indices = self.df.index.to_list()
        if len(all_indices) < num_samples:
            random_indices = all_indices
        else:
            random_indices = random.sample(all_indices, k = num_samples)

        return random_indices

    def __iter__(self):
        """
        Returns an iterator that yields random indices for n-shot
        sampling.
        """
        # Create a copy to avoid modifying the original list
        indices = self.class_indices.copy()
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        """Returns the total number of samples in the sampler."""
        return len(self.class_indices)  # Return actual length of indices

class NShotStartifiedSampler(Sampler):
    """PyTorch Sampler for creating N-shot MultiClass datasets for 
    few-shot prompt learning based methods.
    """
    def __init__(
        self,
        dataset: CXRDatasets,
        n_shot: int = 2,
        random_seed: int = 42
    ):
        self.dataset = dataset
        self.n_shot = n_shot
        self.random_seed = random_seed
        self.num_classes = len(dataset.cxr_labels)

        self.df: pd.DataFrame = dataset.df[
            list(dataset.cxr_labels)
        ].copy(deep=True)

        self.class_indices = self._get_stratified_indices()
        # print(f"Stratified Sampler initialized with {self.n_shot} shots")

    def _get_stratified_indices(self):
        """Get stratified indices for n-shot sampling."""

        pathology_cols = list(self.dataset.cxr_labels)
        labels = self.df[pathology_cols].values
        train_size = self.n_shot * self.num_classes
        total_samples = len(self.df)
        # print(
        # f"Total samples: {total_samples},"
        # f" Train size: {train_size}, Num classes: {self.num_classes}"
        # )

        if train_size >= total_samples:
            raise ValueError(
                f"Requested train_size ({train_size}) is greater than or"
                f" equal to available samples ({total_samples})."
            )

        # Ensure both values are provided
        test_size = total_samples - train_size

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            test_size=test_size, # type: ignore
            random_state=self.random_seed
        )
        train_strat_idx = next(msss.split(self.df, labels))[0]
        train_strat_idx = train_strat_idx.tolist()

        return train_strat_idx

    def __iter__(self):
        """
        Returns an iterator that yields stratified indices
        for n-shot sampling.
        """
        indices = self.class_indices.copy()
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        """Returns the total number of samples in the sampler."""
        return len(self.class_indices) # Return actual length of indices

class VisionSampler(Sampler):
    def __init__(
            self,
            dataset: CXRDatasets,
            n_shot: int = 2,
            random_seed: int = 42
        ):
        self.dataset = dataset
        self.num_classes = len(dataset.cxr_labels)
        self.random_seed = random_seed

        self.df = dataset.df[
            list(dataset.cxr_labels)
        ].copy(deep=True)
        self.class_indices = self._get_random_indices()
        print(f"Vision Sampler initialized with {len(self.class_indices)} samples")

    def _get_random_indices(self):
        """ Get random indices for n-shot sampling.
        """
        all_indices = self.df.index.to_list()
        return all_indices

    def __iter__(self):
        """
        Returns an iterator that yields random indices for n-shot
        sampling.
        """
        # Create a copy to avoid modifying the original list
        indices = self.class_indices.copy()
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        """Returns the total number of samples in the sampler."""
        return len(self.class_indices)  # Return actual length of indices