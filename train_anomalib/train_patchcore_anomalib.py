import os
# Set PyTorch memory management environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
torch.set_float32_matmul_precision('medium')

# Add memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# GPU memory management - limit to available memory
if torch.cuda.is_available():
    # Get available GPU memory in GB
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU memory: {gpu_memory:.2f} GB")
    
    # Set memory fraction to use (e.g., 80% of available memory)
    memory_fraction = 0.8
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Set memory pool settings
    torch.cuda.empty_cache()

from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import Patchcore
from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine
import argparse
from torchvision.transforms import v2

# follow the notebook https://github.com/openvinotoolkit/anomalib/blob/main/notebooks/200_models/201_fastflow.ipynb


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--name_normal_dir', type=str)
    parser.add_argument("--data_augmentation", type=str2bool, nargs='?', const=True, default=False, help="Apply data augmentation during train")
    parser.add_argument('--name_wandb_experiment', type=str)
    opt = parser.parse_args()

    dataset_root = opt.dataset_root
    name_wandb_experiment = opt.name_wandb_experiment
    name_normal_dir = opt.name_normal_dir
    data_augmentation = opt.data_augmentation
    name = opt.name

    # Define a list of transformations you want to apply to your data
    transformations_list = [# geometric transformations
                            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                            v2.RandomAffine(degrees=(0, 1), scale=(0.9, 1.2)),
                            # pixel transformations
                            v2.ColorJitter(brightness=(0.6, 1.6)),
                            v2.ColorJitter(contrast=(0.6, 1.6)),
                            v2.ColorJitter(saturation=(0.6, 1.6)),
                            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.3, 0.3))
                            ]

    transforms = None
    if data_augmentation:
        transforms = v2.RandomApply(transformations_list, p=0.8)

    datamodule = Folder(
        name=name,
        root=dataset_root,
        normal_dir=name_normal_dir,
        abnormal_dir="abnormal",
        task=TaskType.CLASSIFICATION,
        seed=42,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.5,
        train_transform=transforms,
        train_batch_size=1,  # Reduced from 4 to 1
        eval_batch_size=1
    )

    model = Patchcore(coreset_sampling_ratio=0.01)  # Reduced from 0.05 to 0.01

    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="image_AUROC",
            save_last=True,
            verbose=True,
            auto_insert_metric_name=True,
            every_n_epochs=1,
        )
    ]

    wandb_logger = AnomalibWandbLogger(project="image_anomaly_detection",
                                       name=name_wandb_experiment)

    engine = Engine(
        max_epochs=100,
        callbacks=callbacks,
        pixel_metrics="AUROC",
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        logger=wandb_logger,
        task=TaskType.CLASSIFICATION,
        accumulate_grad_batches=4,  # Add gradient accumulation
        precision="16-mixed",  # Use mixed precision to save memory
    )

    print("Fit...")
    engine.fit(datamodule=datamodule, model=model)

    print("Test...")
    engine.test(datamodule=datamodule, model=model)

    #export in torch
    print("Export weights...")
    path_export_weights = engine.export(export_type=ExportType.TORCH,
                                        model=model)

    print("path_export_weights: ", path_export_weights)