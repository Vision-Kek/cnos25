# cnos25

cnos25 follows the [CNOS](https://github.com/nv-nguyen/cnos/) proposal-then-match pipeline.
It uses 
[YOLOE](https://github.com/THU-MIG/yoloe/) as the proposal model and 
[dinov3](https://github.com/facebookresearch/dinov3/) descriptors for matching.

Besides result reproduction, this repo might provide some useful code for you:
* Evaluation tools - which help you to keep track of your experiments
* Visualization tools - nice to get started
* Handling of Hot3D dataset  - which is a bit tricky with its two Aria/Quest3 streams

## Setup
### Datasets
Currently, the three datasets in BOP-H3 are explicitly supported.
When [downloading](https://bop.felk.cvut.cz/datasets/), you can skip all training folders as cnos is training-free and makes no use of them.

Set then `root_dir: /path/prefix/to/bop` in the first line of [bop.yaml](configs/data/bop.yaml). Your dataset(s) `hopev2`,`hot3d`,`handal` are expected to be in a `datasets` dir located in this `root_dir`,
e.g. `/path/prefix/to/bop/datasets/hopev2`.


### Installation
Install the project and its dependencies:
```commandline
conda create -n cnos25 python=3.10
pip install -e .
```

### Checkpoints
1. YOLOE-11l-seg:
[Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt) and set `ultralyticsmodel.model: /path/to/your/yoloe-11l-seg.pt` in [yoloe.yaml](configs/model/proposal/yoloe.yaml).
2. Dinov3 ViT-L/16:
   1. [Download](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and set `hubmodel.weights: /path/to/your/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` in [dinov3.yaml](configs/model/descriptor/dinov3.yaml). 
   2. Clone [dinov3](https://github.com/facebookresearch/dinov3). Set `repo_or_dir` to the dinov3 repo dir.  

## Run it
Following the [BOP workflow](https://bop.felk.cvut.cz/static/img/6d_object_pose_estimation.jpg), there are two stages,
onboarding and inference.

###  1. Onboarding Stage
Extract the reference (=template) descriptors from the onboarding data:
```commandline
python -m src.scripts.extract_template_descriptors dataset_name=hopev2
```
The corresponding config file is [extract_templates.yaml](configs/extract_templates.yaml).

<details><summary>Details</summary>
Descriptors are stored by default in a folder called `descriptors` created in `onboarding_static` of the selected dataset. 
</details>

### 2. Inference Stage:
Predict boxes and segmentations: 
```commandline
python run_inference.py dataset_name=hopev2 split=test
```
The corresponding config file is [run_inference.yaml](configs/run_inference.yaml).

### Additional features
WIP. Please note that the repo is currently WIP.

## Acknowledgement
The code is adapted from [CNOS](https://github.com/nv-nguyen/cnos/). The two models used are
[YOLOE](https://github.com/THU-MIG/yoloe/) and [dinov3](https://github.com/facebookresearch/dinov3/).

## Contact
If you have any question, feel free to create an issue or contact me at jherzog@zju.edu.cn.