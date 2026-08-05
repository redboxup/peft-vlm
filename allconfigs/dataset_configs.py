import dataclasses
from typing import Tuple
from dataclasses import dataclass


@dataclass
class DatasetConf:
    datasetname: str = dataclasses.field(init=False)
    cxr_labels: Tuple[str, ...] = dataclasses.field(init=False)
    txt_filepath: str = dataclasses.field(init=False)
    train_images_filepath: str = dataclasses.field(init=False)
    train_labels_filepath: str = dataclasses.field(init=False)
    val_images_filepath: str = dataclasses.field(init=False)
    val_labels_filepath: str = dataclasses.field(init=False)
    test_labels_filepath: str = dataclasses.field(init=False)
    test_images_filepath: str = dataclasses.field(init=False)
    predictions_dir: str = dataclasses.field(init=False)
    cache_dir: str = dataclasses.field(init=False)
    verify_dataloaders: bool = False


@dataclass
class CheXpertConf(DatasetConf):
    datasetname: str = 'chexpert'
    # class labels
    cxr_labels: Tuple[str, ...] = (
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    )
    # data
    train_images_filepath: str = (
        'data/preprocessed/chexpert/chexpertchestxrays-u20210408'
    )
    train_labels_filepath: str = (
        'data/preprocessed/chexpert/modified_train.csv'
    )
    val_images_filepath: str = (
        'data/preprocessed/chexpert/chexpertchestxrays-u20210408/'
    )
    val_labels_filepath: str = (
        'data/preprocessed/chexpert/modified_valid.csv'
    )
    test_images_filepath: str = (
        'data/preprocessed/chexpert/test/chexlocalize/CheXpert/'
    )
    test_labels_filepath: str = (
        'data/preprocessed/chexpert/test/chexlocalize/CheXpert/'
        'ground_truth.csv'
    )
    zeroshot_results_dir: str = (
        "/mnt/6a7d1ed7-f3e3-470f-aec0-5f6cd7e17ea5/anudeep/CLIPOOD/experiments_new/chexpert/zeroshot_0/multiclass/exp_00000000/results/ID_test_auc.csv"
    )


@dataclass
class VinCXRConf(DatasetConf):
    datasetname: str = 'vincxr'
    # class labels
    cxr_labels: Tuple[str, ...] = (
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema',
        'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity',
        'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion',
        'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis',
        'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia',
        'Tuberculosis', 'Other disease', 'No finding'
    )
    # data
    train_images_filepath: str = 'data/preprocessed/vincxr/train'
    train_labels_filepath: str = (
        'data/preprocessed/vincxr/image_labels_train_split.csv'
    )
    val_images_filepath: str = 'data/preprocessed/vincxr/train'
    val_labels_filepath: str = (
        'data/preprocessed/vincxr/image_labels_val_split.csv'
    )
    test_images_filepath: str = 'data/preprocessed/vincxr/test'
    test_labels_filepath: str = (
        'data/preprocessed/vincxr/image_labels_test.csv'
    )
    zeroshot_results_dir: str = (
        "/mnt/6a7d1ed7-f3e3-470f-aec0-5f6cd7e17ea5/anudeep/CLIPOOD/experiments_new/vincxr/zeroshot_0/multiclass/exp_00000000/"
        "results/ID_test_auc.csv"
    )


@dataclass
class NIHCXRConf(DatasetConf):
    datasetname: str = 'nihcxr'
    # class labels
    cxr_labels: Tuple[str, ...] = (
        'Pneumothorax', 'Pneumonia', 'Cardiomegaly', 'Consolidation', 'Nodule',
        'Pleural_Thickening', 'Hernia', 'Fibrosis', 'No Finding', 'Edema',
        'Emphysema', 'Infiltration', 'Mass', 'Effusion', 'Atelectasis'
    )
    # data
    train_images_filepath: str = 'data/preprocessed/nihcxr/all_images'
    train_labels_filepath: str = (
        'data/preprocessed/nihcxr/modified_train.csv'
    )
    val_images_filepath: str = 'data/preprocessed/nihcxr/all_images'
    val_labels_filepath: str = 'data/preprocessed/nihcxr/modified_val.csv'
    test_images_filepath: str = 'data/preprocessed/nihcxr/all_images'
    test_labels_filepath: str = 'data/preprocessed/nihcxr/modified_test.csv'
    zeroshot_results_dir: str = (
        "/mnt/6a7d1ed7-f3e3-470f-aec0-5f6cd7e17ea5/anudeep/CLIPOOD/experiments_new/nihcxr/zeroshot_0/multiclass/exp_00000000/"
        "results/ID_test_auc.csv"
    )


@dataclass
class MIMIC_CXRConf(DatasetConf):
    datasetname: str = 'mimic-cxr'
    # hdf5_filepath: str = f'data/preprocessed/preprocessed_cxr/{datasetname}_test.h5'
    txt_filepath: str = f'data/preprocessed/mimic_cxr/'


@dataclass
class OODDatasetConf:
    dataset_dict: dict = dataclasses.field(default_factory=lambda: {
        # check total images
        'mura': 'data/preprocessed/mura/MURA-v1.1/valid',
        # total images: 9112
        'irma': 'data/irma_preprocessed/irma_proprocessed_without_ID',
        # total images: 12612
        'boneage': 'data/preprocessed/boneage/boneage-training-dataset',
    })
