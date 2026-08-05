import shutil
import pydicom
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from PIL import Image, ImageOps
from multiprocessing import Pool
from typing import Tuple, Callable
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


class ImagePipeline:
    def __init__(self, rawdatapath: str, outdatapath: str):
        self.path: Path

        self.rawpath = Path(rawdatapath)
        if not self.rawpath.exists():
            error_desc = f"Error: The path {rawdatapath} doesn't exist"
            raise FileNotFoundError(error_desc)

        self.outpath = Path(outdatapath)
        self.outpath.mkdir(exist_ok=True, parents=True)

    def load_image(self, imgpath: Path):
        self.path = imgpath
        self.img = Image.open(imgpath)
        return self

    def load_dicom(self, dcmpath: Path):
        self.path = dcmpath
        image_dcm = pydicom.dcmread(dcmpath)
        image_array = (
            image_dcm
            .pixel_array
            .astype(np.float32)
        )
        image_array -= image_array.min()
        image_array /= image_array.max()
        image_array *= 255.0
        image_array = (
            np.clip(image_array, 0, 255)
            .astype(np.uint8)
        )
        image_array = Image.fromarray(image_array)
        # check if image needs to be inverted
        if image_dcm.PhotometricInterpretation == "MONOCHROME1":
            image_array = ImageOps.invert(image_array)

        # histogram equalization for contrast enhancement
        self.img = ImageOps.equalize(image_array)
        return self

    def resize_img(self, size: Tuple[int, int]):
        self.img = self.img.resize(size)
        return self

    def save_img(self):
        relative_path = self.path.relative_to(self.rawpath)
        save_img_dir = self.outpath / relative_path.parent
        save_img_dir.mkdir(parents=True, exist_ok=True)

        save_img_name = save_img_dir / f"{self.path.stem}.png"
        self.img.save(save_img_name)


def preprocess_func(
    img_path: Path,
    loader_func: Callable[[Path], ImagePipeline],
    size: Tuple[int, int],
):
    try:
        return (
            loader_func(img_path)
            .resize_img(size=size)
            .save_img()
        )
    except Exception as e:
        print(f"WARNING: Failed to process {img_path}: {e}")
        return None


parser = argparse.ArgumentParser('Process raw images')

parser.add_argument(
    '--raw-folder-path',
    type=str,
    required=True,
    help='path to raw dataset'
)
parser.add_argument(
    '--out-folder-path',
    type=str,
    required=True,
    help='path to save processed dataset'
)
parser.add_argument(
    '--resize',
    type=int,
    default=320,
    help='resize resolution of images'
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=16,
    help='Assigns threads to parallelize preprocessing'
)

if __name__ == '__main__':
    args = parser.parse_args()
    raw_path = Path(args.raw_folder_path)
    out_path = Path(args.out_folder_path)

    pipeline = ImagePipeline(
        rawdatapath=args.raw_folder_path,
        outdatapath=args.out_folder_path,
    )

    raw_subdirs = [
        entry for entry in Path(args.raw_folder_path).iterdir()
        if entry.is_dir()
    ]
    preprocessed_subdirs = [
        entry.name for entry in Path(args.out_folder_path).iterdir()
        if entry.is_dir()
    ]

    unprocessed_subdirs = [
        entry for entry in raw_subdirs
        if entry.name not in preprocessed_subdirs
    ]

    if preprocessed_subdirs:
        print(f"Preprocessing {len(unprocessed_subdirs)} new directories")
    else:
        print("No previously processed directories found.")

    for directory in unprocessed_subdirs:
        if directory.name == 'vincxr':
            all_img_paths = [path for path in directory.rglob('*.dicom')]
            func = partial(
                preprocess_func,
                loader_func=pipeline.load_dicom,
                size=(args.resize, args.resize),
            )

        elif directory.name in ['nihcxr', 'boneage', 'mura', 'irma']:
            all_img_paths = [path for path in directory.rglob('*.png')]
            func = partial(
                preprocess_func,
                loader_func=pipeline.load_image,
                size=(args.resize, args.resize),
            )

        elif directory.name == 'chexpert':
            all_img_paths = [
                path for path in directory.rglob('*_frontal.jpg')
            ]
            func = partial(
                preprocess_func,
                loader_func=pipeline.load_image,
                size=(args.resize, args.resize),
            )

        else:
            print(f"Unsupported Dataset: {directory.name}")
            continue

        # multiprocessing for speed-up
        # all_img_paths = all_img_paths[:100]
        with Pool(args.num_workers) as pool:
            for _ in tqdm(
                pool.imap(func, all_img_paths),
                total=len(all_img_paths),
                desc=f"Preprocessing directory: {directory.name}"
            ):
                pass

        if directory.name in ['vincxr', 'nihcxr', 'chexpert']:
            # copy *.csv, *.txt files for id-datasets
            other_files = (
                list(directory.rglob('*.csv')) +
                list(directory.rglob('*.txt')) +
                list(directory.rglob('*.xlsx'))
            )

            for src_filepath in other_files:
                relative_path = src_filepath.relative_to(raw_path)
                destination_path = out_path / relative_path
                destination_path.parent.mkdir(exist_ok=True, parents=True)

                shutil.copy2(
                    str(src_filepath),
                    str(destination_path),
                )

            # create a val-split for vincxr
            if directory.name == 'vincxr':
                train_df = pd.read_csv(
                    str(
                        out_path /
                        'vincxr' /
                        'image_labels_train.csv'
                    )
                )

                # modify col name `Other diseases` -> `Other disease`
                # To ensure uniformity in train & test csv files
                train_df.rename(
                    columns={'Other diseases': 'Other disease'},
                    inplace=True
                )

                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.2,  # type: ignore
                    random_state=42,
                )

                X = train_df[['image_id']]
                y = train_df[train_df.columns[2:]]

                for train_idx, val_idx in msss.split(X, y):
                    train_split = (
                        train_df
                        .iloc[train_idx]
                        .reset_index(drop=True)
                    )
                    val_split = (
                        train_df
                        .iloc[val_idx]
                        .reset_index(drop=True)
                    )

                    _path = out_path / 'vincxr'
                    train_csv_path = _path / 'image_labels_train_split.csv'
                    val_csv_path = _path / 'image_labels_val_split.csv'

                    train_split.to_csv(train_csv_path, index=False)
                    val_split.to_csv(val_csv_path, index=False)

            # handle unknown/unsure label in chexpert
            if directory.name == 'chexpert':
                _path = (
                    out_path /
                    'chexpert' /
                    'chexpertchestxrays-u20210408' /
                    'CheXpert-v1.0 batch 1 (validate & csv)'
                )
                train_df = pd.read_csv(str(_path / 'train.csv'))
                valid_df = pd.read_csv(str(_path / 'valid.csv'))

                # Filter out lateral-view, only keep frontal views
                train_df = train_df[train_df['Frontal/Lateral'] == 'Frontal']
                valid_df = valid_df[valid_df['Frontal/Lateral'] == 'Frontal']

                # 1 -> finding present, 0 -> finding absent
                # -1 -> finding uncertain, NaN -> Missing in report
                train_df = train_df.fillna(0).replace(-1, 0)
                valid_df = valid_df.fillna(0).replace(-1, 0)

                train_csv_path = out_path / 'chexpert' / 'modified_train.csv'
                valid_csv_path = out_path / 'chexpert' / 'modified_valid.csv'

                train_df.to_csv(train_csv_path, index=False)
                valid_df.to_csv(valid_csv_path, index=False)

                # Merge the followings dirs into one
                source_dirs = [
                    Path('CheXpert-v1.0 batch 1 (validate & csv)'),
                    Path('CheXpert-v1.0 batch 2 (train 1)'),
                    Path('CheXpert-v1.0 batch 3 (train 2)'),
                    Path('CheXpert-v1.0 batch 4 (train 3)'),
                ]

                train_target_dir = Path('CheXpert-v1.0/train')
                val_target_dir = Path('CheXpert-v1.0/valid')
                train_target_dir.mkdir(exist_ok=True, parents=True)
                val_target_dir.mkdir(exist_ok=True, parents=True)

                for src_dir in source_dirs:
                    for subdir in src_dir.iterdir():
                        if subdir.is_dir():
                            target = (
                                train_target_dir / subdir.name
                                if 'train' in src_dir.name
                                else val_target_dir / subdir.name
                            )
                            if target.exists():
                                raise FileExistsError(
                                    f"Target {target} already exists"
                                )
                            shutil.move(str(subdir), target)

            # make train-val split for nihcxr
            if directory.name == 'nihcxr':
                df = pd.read_csv(
                    str(directory / 'Data_Entry_2017.csv'),
                    index_col='Image Index     ',
                )

                # creating `findings` column, consisting of list of
                # findings,
                # eg: 'Cardiomegaly | Pneumonia' will become an element
                # ['Cardiomegaly', 'Pneumonia']
                df['findings'] = (
                    df[df.columns[0]]
                    .str.strip()  # remove unnecessary whitespace
                    .str.split('|')
                )

                # multi-label encoding
                mlb = MultiLabelBinarizer()
                new_df = pd.DataFrame(
                    mlb.fit_transform(df['findings']),
                    columns=mlb.classes_,
                    index=df.index
                )

                # split df to train_val_df & test_df
                train_idx = (
                    pd.read_csv(
                        str(directory / 'train_val_list.txt'),
                        header=None,
                    )
                    .squeeze()
                    .to_list()[1:]  # type: ignore
                )
                test_idx = (
                    pd.read_csv(
                        str(directory / 'test_list.txt'),
                        header=None,
                    )
                    .squeeze()
                    .to_list()[1:]  # type: ignore
                )
                train_val_df = new_df.loc[train_idx]
                test_df = new_df.loc[test_idx]

                # split train_val_df into train & val
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.2,  # type: ignore
                    random_state=42,
                )

                X = train_val_df.index
                y = train_val_df[train_val_df.columns]

                _path = out_path / 'nihcxr'
                for train_idx, val_idx in msss.split(X, y):
                    (
                        train_val_df
                        .iloc[train_idx]
                        .reset_index()
                        .to_csv(str(_path / 'modified_train.csv'))
                    )
                    (
                        train_val_df
                        .iloc[val_idx]
                        .reset_index()
                        .to_csv(str(_path / 'modified_val.csv'))
                    )

                (
                    test_df
                    .reset_index()
                    .to_csv(str(_path / 'modified_test.csv'))
                )
                # Consolidate all image_0*/images/ directories into a
                # single all_images/ directory. This is necessary
                # because the dataset does not use relative paths for
                # images, and repeatedly searching for image files
                # across multiple directories would be inefficient.
                target_dir = _path / 'all_images'
                target_dir.mkdir(exist_ok=True)
                for path in _path.glob('images_0*/images/*.png'):
                    path.rename(
                        str(target_dir / path.name),
                    )
