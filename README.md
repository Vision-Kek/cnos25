# cnos25
Tempalate-based novel object detection and segmentation.

cnos25 follows the [CNOS](https://github.com/nv-nguyen/cnos/) proposal-then-match pipeline.
It uses 
[YOLOE](https://github.com/THU-MIG/yoloe/) as the proposal model and 
[dinov3](https://github.com/facebookresearch/dinov3/) descriptors for matching.

Besides result reproduction, this repo might provide some useful code for you:
* Evaluation tools - which help you to keep track of your experiments
* Visualization tools - nice to get started
* Handling of Hot3D dataset  - which is a bit tricky with its two Aria/Quest3 streams

## Setup
All paths you need to set up are in [local.yaml](configs/local.yaml).
### Datasets
Currently, the three datasets in BOP-H3 are explicitly supported.
When [downloading](https://bop.felk.cvut.cz/datasets/), you can skip all training folders as cnos is training-free and makes no use of them.

<details><summary>File Tree</summary>


```bash
bop_data_root/
├── handal/
│   ├── test_metaData.json
│   ├── test_targets_bop24.json
│   ├── onboarding_static
│   │   ├── obj_00000xx/
│   │   └── ...
│   ├── val/
│   │   ├── 000001/
│   │   └── ...
│   ├── test/
│   │   ├── 000011/
│   │   └── ...
│   └── ...
├── hopev2/
│   └──  same as handal
├── hot3d/
│   ├── clip_definitions.json
│   ├── clip_splits.json
│   ├── test_targets_bop24.json
│   ├── onboarding_static/
│   │   ├── obj_00000xx
│   │   └── ...
│   ├── test_aria/
│   │   ├── clip-003xxx.tar
│   │   └── ...
│   ├── test_quest3/
│   │   ├── clip-001xxx.tar
│   │   └── ...
└── └── ...
```

</details>

After downloading, set `bop_data_root` in `local.yaml`


### Installation
Install the project and its dependencies:
```commandline
conda create -n cnos25 python=3.10
pip install -e .
```

### Checkpoints
1. yoloe-11l-seg:
[Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt). Set `yoloe_checkpoint` in `local.yaml`.
2. dinov3 ViT-L/16:
   1. [Download](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Set `dinov3_checkpoint` in `local.yaml`. 
      
   2. [Clone](https://github.com/facebookresearch/dinov3). Set `dinov3_repo` in `local.yaml`.

## Run it
Following the [BOP workflow](https://bop.felk.cvut.cz/static/img/6d_object_pose_estimation.jpg), there are two stages,
onboarding and inference.

###  1. Onboarding Stage
Extract the reference (=template) descriptors from the onboarding data:
```commandline
python -m src.scripts.extract_template_descriptors dataset_name=hopev2
```
The corresponding config file is [extract_templates.yaml](configs/extract_templates.yaml).

<details><summary>cache path</summary>

Descriptors are stored by default in a folder called `descriptors` created in `onboarding_static` of the selected dataset. 

</details>

### 2. Inference Stage
Predict boxes and segmentations: 
```commandline
python run_inference.py dataset_name=hopev2 split=test
```
The corresponding config file is [run_inference.yaml](configs/run_inference.yaml).

<details><summary>auto-downloads</summary>

On the first run, ultralytics will automatically install a package `clip` and download `mobileclip_blt.ts` (572MB),
which are required for textual prompting of YOLOE.

</details>

<details><summary>bop_toolkit troubleshooting</summary>

* `datetime.UTC` error in `bop_toolkit_lib/misc.py` - Fix: Change to `datetime.timezone.utc`.
* `COCO` error in `scripts/eval_bop22_coco.py` - Fix: Replace `cocoGt = COCO(dataset_coco_ann)` with:
    ```python 
    _f='/tmp/dataset_coco_ann.json'
    with open(_f,'w') as f:
        json.dump(dataset_coco_ann, f)
    cocoGt = COCO(_f)
    ```
Reason: Deprecated calls to `datetime` and `pycocotools` in `bop_toolkit_lib`.

</details>

### Additional features
Coming soon.

## Acknowledgement
The code is adapted from [CNOS](https://github.com/nv-nguyen/cnos/). The two models used are
[YOLOE](https://github.com/THU-MIG/yoloe/) and [dinov3](https://github.com/facebookresearch/dinov3/).

## Contact
If you have any question or feature request, feel free to create an issue or contact me at jherzog@zju.edu.cn.
