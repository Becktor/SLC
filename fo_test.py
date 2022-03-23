import fiftyone as fo
import os

name = "ue_coco"
for x in ['train', 'val']:
    dataset_dir = os.path.join(r'Q:\yolo\ue_gen', x)
    masks = os.path.join(dataset_dir, "mask")
    imgs = os.path.join(dataset_dir, "img")
    json = os.path.join(dataset_dir, f"{x}.json")
    name = f"{name}_{x}"

    # Create the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=imgs,
        labels_path=json,
        name=name,
    )

    export_dir = r"Q:\yolo\ue_gen\yolo"
    #export_dir = os.path.join(tmp_dir, x)
    #if not os.path.exists(export_dir):
    #    os.makedirs(export_dir)
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=x,
        label_field="ground_truth",
    )
    # View summary info about the dataset
    # print(dataset)

    # Print the first few samples in the dataset
    # print(dataset.head())
