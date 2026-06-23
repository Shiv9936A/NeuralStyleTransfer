import fiftyone as fo  # This is the line you are missing
import fiftyone.zoo as foz

# Your existing code
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=1000
)

dataset.export(
    export_dir="data",
    dataset_type=fo.types.ImageDirectory
)