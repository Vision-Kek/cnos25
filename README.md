# cnos25
<a href="https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3">
    <img src="https://img.shields.io/badge/BOP Challenge 2025-8A2BE2" />
</a>

**[2025.11.16]** Winner of **The Best Fast Method**, **The Best Open Source Method** Awards, Track 7: Model-free 2D detection of unseen objects on BOP-H3, 10th International Workshop on Recovering 6D Object Pose, ICCV 2025.

**[2025.12.16]** Add custom image and visual prompt demo.

---

Template-based novel object detection and segmentation.

<details><summary>About </summary>

cnos25 follows the [CNOS](https://github.com/nv-nguyen/cnos/) propose-then-match pipeline.
It uses YOLOE as the proposal model and dinov3 descriptors for matching.

Besides result reproduction, this repo might provide some useful code for you:
* Handling of Hot3D dataset  - which is a bit tricky with its two rgb/gray streams, Aria/Quest3 devices and clip_utils dependency
* Evaluation tools - which help you to keep track of your experiments
* Visualization tools and demos - nice to get started

</details>

---

## 🔧 Setup

### Installation
Install the project and its dependencies:
```commandline
pip install -e .
```

### Checkpoints
All paths you need to set up are in [configs/local.yaml](configs/local.yaml).
1. yoloe-11l-seg:
[Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt). Set `yoloe_checkpoint:` in `local.yaml`.
2. dinov3 ViT-L/16:
   1. [Download](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Set `dinov3_checkpoint:` in `local.yaml`. 
      
   2. [Clone](https://github.com/facebookresearch/dinov3). Set `dinov3_repo:` in `local.yaml`.


## 🚀 Quick start
### Custom Images & Visual Prompting
I noticed a great interest in visual prompting for detection & segmentation.

While SAM3 and YOLOE attempt to support this to some extent, you might have noticed that its usefulness is very limited.

The CNOS pipeline is actually a solution for that!
It supports cross-image (unlike SAM3) and multi-reference (unlike YOLOE).
[Try in this notebook](custom_images.ipynb) with your own images!

![](assets/custom_img_res_viz.png)

### With BOP Datasets
Following the [BOP workflow](https://bop.felk.cvut.cz/static/img/6d_object_pose_estimation.jpg), there are two stages,
*onboarding* and *inference* - check out the [onboarding notebook](onboarding_demo.ipynb) and the [inference notebook](inference_demo.ipynb):

![demo.png](assets/demo.png)

<details><summary>BOP Dataset Setup </summary>

Currently, the three datasets in BOP-H3 are explicitly supported.
When [downloading](https://bop.felk.cvut.cz/datasets/), you can skip all training folders as cnos is training-free and makes no use of them.
You can refer to [this script](https://github.com/nv-nguyen/cnos/blob/main/src/scripts/download_modelfree_onboarding_bop_h3.py) for downloading, unzipping and renaming.


After downloading, set `bop_data_root:` in `local.yaml`.

```bash
bop_data_root/
├── handal/
│   ├── test_metaData.json
│   ├── test_targets_bop24.json
│   ├── onboarding_static
│   │   ├── obj_00000xx_up/
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
│   ├── onboarding_static/ # -> object_ref_aria_static 
│   │   ├── obj_00000xx_up.tar
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

## ▶️ Full run

###  1. Onboarding Stage
Extract the reference (=template) descriptors from the onboarding data:
```commandline
python -m src.scripts.extract_template_descriptors dataset_name=hopev2
```
The corresponding config file is [extract_templates.yaml](configs/extract_templates.yaml).

<details><summary>cache path</summary>

Descriptors are stored by default in `onboarding_static/descriptors` of the selected dataset.
The default output file name is `${model_name}_descriptors.pt`.
Change it by passing `out_file=foo.pt`.

</details>

<details><summary>hydra troubleshooting</summary>

1. Make sure you have all paths correctly set in `local.yaml` and the dataset [file tree](#datasets) matches.
2. You can override all config params from CLI. For example, if there are issues during template extraction, you can have a fast test run with only 6 instead of 100 samples per object:
    ```
    python -m src.scripts.extract_template_descriptors data.reference_dataloader.num_imgs_per_obj=6 out_file=dummy.pt dataset_name=...
    ```
3. Instead of running everything as a single python script, run modularly using our provided [notebooks](). It might be easier to spot the exact issue there.

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

* `datetime.UTC` error in `bop_toolkit_lib/misc.py` - Fix: Change to `datetime.timezone.utc` [#203](https://github.com/thodan/bop_toolkit/commit/b9fefca50e7cf79a007d9e1603888f127fe12fac).
* `COCO` error in `scripts/eval_bop22_coco.py` - Fix: Replace `cocoGt = COCO(dataset_coco_ann)` with:
    ```python 
    _f='/tmp/dataset_coco_ann.json'
    with open(_f,'w') as f:
        json.dump(dataset_coco_ann, f)
    cocoGt = COCO(_f)
    ```
Reason: Deprecated calls to `datetime` and `pycocotools` in `bop_toolkit_lib`.

</details>

<details><summary>Custom datasets</summary>

You can have a look at the [template for custom datasets](configs/data/custom_datasets.yaml).
Complete it and replace `bop` through it as the `defaults` config option for [onboarding](configs/extract_templates.yaml) and [inference](configs/run_inference.yaml).

</details>

<details><summary>Output dir and files</summary>

Once you have started a run, its results are written into hydra-set `outputs/yyyy-mm-dd/hh-mm-ss_xxxx/`.
You'll find
* a `predictions` folder containing the image-wise predictions as npz files.
* a `nms-{...}.json` that contains the accumulated and postprocessed predictions that can be submitted to the BOP challenge.

</details>

## 📈 Results

[Download .json](https://bop.felk.cvut.cz/method_info/1161), use as better default detections for your 6D pose estimation method.
|         | Proposal-Descriptor | H3    | Hot3D | Hopev2 | Handal |
|---------|---------------------|-------|-------|--------|--------|
| cnos    | FastSAM-dinov2      | 0.340 | 0.373 | 0.343  | 0.304  |
| cnos25  | YOLOE-dinov3        | 0.441 | 0.481 | 0.452  | 0.389  |

Avg. 0.134 sec / image on RTX4090.

<details><summary>Measure AP on validation split</summary>

If you pass `split=val`, resulting AP is directly measured since ground truth is available for the validation set.
This requires two setup steps:
1. Since this invokes the [bop toolkit script](https://github.com/thodan/bop_toolkit/blob/master/scripts/eval_bop22_coco.py),
`git clone https://github.com/thodan/bop_toolkit/` and set `bop_toolkit_repo:` in [local.yaml](configs/run_inference.yaml).
2. Copy the respective `val_targets_bop24.json` from [val_targets/](val_targets) into your `{bop_data_root}/{dataset_name}`.

</details>

<details><summary>Preprocessing for reported results</summary>

Reported results from the [BOP challenge submission](https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3)
have been produced with input images rotated for some datasets, because YOLOE is not optimal on rotated images.
This prepocessing configuration can be set in [yoloe.yaml](configs/model/proposal/yoloe.yaml): 
* For Hot3D, all images are rotated clockwise by 90deg: `rotate_input_images: [ -90 ]`
* For Handal, a batch of 3 images is created: `rotate_input_images: [ 0, 90, -90 ]`. The frame with the highest cumulative confidence will be selected.
* For Hopev2, no such rotation was done: `rotate_input_images: [ ]`

</details>

<details><summary>Visualize results</summary>

Provide `dataset_name`, `split`, and the result `.json` to the following script:
```bash
mkdir viz
python -m src.scripts.visualize_detectron2 dataset_name={...} split={...} input_file=outputs/{...}/nms-{...}.json output_dir=viz
```
The corresponding config file is [run_vis.yaml](configs/run_vis.yaml). Adopted from original cnos.

You can also download any submission file from the BOP website and feed it.
</details>

## Acknowledgement
The code is adapted from [CNOS](https://github.com/nv-nguyen/cnos/). The two models used are
[YOLOE](https://github.com/THU-MIG/yoloe/) and [dinov3](https://github.com/facebookresearch/dinov3/).

## Contact
If you have any question or feature request (support of dynamic/model-based onboarding, other datasets, custom images etc.) feel free to create an issue or contact me at jherzog@zju.edu.cn.
