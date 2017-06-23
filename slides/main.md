name: inverse
class: center, middle, inverse
layout: true

---

class: titlepage, no-number

# ChainerCV: a Library for Deep Learning <br /> in .red[C]omputer .red[V]ision

## .author[Shunta Saito]
### .small[.white[Teatime Tech Talk @PFN, Jun 22th, 2017]]
### .small[This slide was originally created by .yellow[Yusuke Niitani] for Chainer Meetup #05]
### .small[The original slide is: https://yuyu2172.github.io/chainercv-chainer-meetup]
### .small[This slide is: https://mitmul.github.io/chainercv-tech-talk]

---
layout: false

## What is ChainerCV?

.center.img[![chainercv screenshot](images/screenshot.png)]

* An add-on package built on top of Chainer for computer vision
* Developed in this repo:  [https://github.com/chainer/chainercv](https://github.com/chainer/chainercv)
* Works with `chainer>=2.0.0` (also works for Chainer v1)
* MIT License
* Developed since late February 2017

---

## ChainerCV Contributors

* Yusuke Niitani ([@yuyu2172](https://github.com/yuyu2172))
* Toru Ogawa ([@hakuyume](https://github.com/hakuyume))
* Shunta Saito ([@mitmul](https://github.com/mitmul))
* Masaki Saito ([@rezoo](https://github.com/rezoo))
* and more...

---
class: split-30

## Why we develop ChainerCV?

#### Make *running* and *training* deep-learning models easier for Computer Vision tasks

<!--
* Network implementations and training scripts
  * Object Detection (Faster R-CNN, SSD)
  * Semantic Segmentation (SegNet)
* Variety of tool sets
* Dataset Loader (e.g. PASCAL VOC) and data-augmentation tools
* Visualization
* Evaluation
-->

<!--.center.img-33[![Right-algined text](images/faster_rcnn_image_000008.png)]-->
.column[
- Technologies in computer vision has been dramatically advanced by deep learning
- But some models are not easy for some people who are not specialist in computer vision to train them with their data, or even just apply them to new dataset.
]
.column[
.center.img-60[![](images/example_outputs_first_page_1.png)]
]

---
class: split-333

## Goals of ChainerCV

1. .large[Easy-to-use Implementation]
2. .large[Provide Tools for Training Networks]
3. .large[Efforts on Reproducible Research]
<!-- 4. Comparison and Conclusions -->

---

template: inverse

# .x-large[1: Easy-to-use Implementation]

---

## Using unofficial implementation is sometimes tough...

* Interface is not so clear (e.g. How to run with my data?)
* Installation failure (e.g. Unmaintained Caffe)
* .red[Actually, even if it's official, using it is sometimes tough at the same reasons]

## ChainerCV aims at solving these issues

1. Easy installation (`pip install chainercv`)
2. Well tested and documented like Chainer
3. Unified interface (next slide)

<!-- because their instructions are unclear -->

---

## Unified Interface

#### Provide the same interface for instantiating different models

```python
model = FasterRCNNVGG16()  # Start from scrach
# Use pre-trained model
model = FasterRCNNVGG16(pretrained_model='voc07')

# Same arguments are prepared for different models
model = SSD300(pretrained_model='voc0712')
model = SegNet(pretrained_model='camvid')
```

--

#### Provide the same interface for different tasks

```python
bboxes, labels, scores = model.predict(imgs)  # Prediction with detection models

labels = model.predict(imgs)  # Prediction with segmentation models
```

---
class: split-30

## Inside of `predict` for Detection Models

.column[
Internally, `predict` does ...
1. Preprocess images (e.g. mean subtraction and resizing)
2. Forward the images through the network
3. Post-processing outputs, e.g., removing overlapping bounding boxes
]
.column[
.center.img-66[![](images/predict_doc.png)]
]

---
class: split-40

## Potential applications

#### Easy-to-use implementation enables to be a building block for other networks

.column[
.small[
- For example, a scene graph generation method depends on the object detection results came from Region Proposal Network (RPN) which was a part of Faster R-CNN model.
- Such a cascaded model can't produce even the first result without implementing the lower structures.
- So providing the easy-to-use implementation of general object detection methods etc. may accelerate such more higher level research.
]
]
.column[
.center.img-60[![scene_graph](images/scene_graph.png)]
.right.x-small[[Scene Graph Generation by Iterative Message Passing. Xu et.al., CVPR2017](https://arxiv.org/abs/1701.02426)]
]

---

template: inverse

# .large[2: Provide Tools for Training Networks]

---

## General Work-flow of Training a Neural Network


* `chainer.training` provides training utilities for general machine learning tasks.
* It is useful, but when we use it in our research, additional tools are sometimes necessary.

<!-- Add a slide on how learning a machine software components -->

.mermaid[graph LR
A[Dataset Loader]-->B[Data Pre-processing]
B-->C[Forward Model]
C-->D[Backprop. and Model Update]
C-->E[Visualization]
C-->F[Evaluation]
]

---

## General Work-flow of Training a Neural Network


* `chainer.training` provides training utilities for general machine learning tasks.
* It is useful, but when we use it in our research, additional tools are sometimes necessary.

<!-- Add a slide on how learning a machine software components -->

.mermaid[graph LR
A[Dataset Loader]-->B[Data Pre-processing]
B-->C[Forward Model]
C-->D[Backprop. and Model Update]
C-->E[Visualization]
C-->F[Evaluation]
style A stroke:#333,stroke-width:4px
style B stroke:#333,stroke-width:4px
style E stroke:#333,stroke-width:4px
style F stroke:#333,stroke-width:4px
]

---

class: middle

.mermaid[graph LR
A[Dataset Loader]-->B[Data Pre-processing]
B-->C[Forward Model]
C-->D[Backprop. and Model Update]
C-->E[Visualization]
C-->F[Evaluation]
style A stroke:#f00,stroke-width:4px
style B stroke:#333,stroke-width:4px
style E stroke:#333,stroke-width:4px
style F stroke:#333,stroke-width:4px
]

---

## Dataset loader

- Similar to dataset loaders in `chainer.datasets` (e.g. MNIST)
- Very easy to start using various public datasets for Chainer projects

```python
from chainercv.datasets import VOCDetectionDataset

*dataset = VOCDetectionDataset(split='trainval', year='2007')
# Access 34th sample in the dataset
img, bbox, label = dataset[34]
```

--
class: split-50

### List of currently supported datasets

.column[
* PASCAL VOC
* CUB-200
* CamVid
]
.column[
* Online Products Dataset
* MNIST
* CIFAR 10
* CIFAR 100
]

---

class: middle

.mermaid[graph LR
A[Dataset Loader]-->B[Data Pre-processing]
B-->C[Forward Model]
C-->D[Backprop. and Model Update]
C-->E[Visualization]
C-->F[Evaluation]
style A stroke:#333,stroke-width:4px
style B stroke:#f00,stroke-width:4px
style E stroke:#333,stroke-width:4px
style F stroke:#333,stroke-width:4px
]

---
class: split-50

## Data Pre-processing: Supported Transform

ChainerCV provides some functions that take images and labels as inputs and apply transforms to them

.column[
* Currently provided transforms for images
    * `center_crop`
    * `pca_lightning`
    * `random_crop`
    * `random_expand`
    * `random_flip`
    * `random_rotate`
    * `ten_crop`
    * etc...
* Transforms for labels such as bounding boxes and keypoints are also provided
]
.column[
.center.img-50[![random_expand](images/mnist_random_expand.png)]
.right.x_small[An example result of `random_expand`]
]
---

## Data Pre-processing: TransformDataset

* An utility to extend an existing dataset enables to apply a user-defined function to each datum.
* This puts together datasets and transforms.

```python
# `dataset` is a dataset for Detection task
def flip_transform(in_data):
    img, bbox, label = in_data
    img, param = random_flip(img, x_flip=True, return_param=True)
    bbox = flip_bbox(bbox, x_flip=param['x_flip'])
    return img, bbox, label

new_dataset = TransformDataset(dataset, flip_transform)
```

- The horizontal flip for an image is randomly performed.
- When the image was flipped, the coordinates of corresponding bounding boxes are also horizontally flipped.

---

class: middle

.mermaid[graph LR
A[Dataset Loader]-->B[Data Pre-processing]
B-->C[Forward Model]
C-->D[Backprop. and Model Update]
C-->E[Visualization]
C-->F[Evaluation]
style A stroke:#333,stroke-width:4px
style B stroke:#333,stroke-width:4px
style E stroke:#f00,stroke-width:4px
style F stroke:#333,stroke-width:4px
]

---

## Visualization

* Visualization utilities for all the data types used in ChainerCV
  * Images
  * Bounding boxes
  * Segmentation labels
  * Keypoints

* Code is built on top of Matplotlib

.center.img-66[![sample_visualization](images/vis_visualization.png)]

---

class: middle

.mermaid[graph LR
A[Dataset Loader]-->B[Data Pre-processing]
B-->C[Forward Model]
C-->D[Backprop. and Model Update]
C-->E[Visualization]
C-->F[Evaluation]
style A stroke:#333,stroke-width:4px
style B stroke:#333,stroke-width:4px
style E stroke:#333,stroke-width:4px
style F stroke:#f00,stroke-width:4px
]

---

## Evaluation

ChainerCV provides tools to calculate some evaluation metrics commonly used in computer vision tasks.

* Semantic Segmentation: IoU
* Object Detection: Average Precision

```python
evaluator = chainercv.extension.DetectionVOCEvaluator(   # Report mAP
        iterator, detection_model)
# `result` contains dictionary of evaluation results
# ex:  result['main/map'] contains mAP
result = evaluator()
```

#### It can be used as `Trainer`'s `Extension` in Chainer

```python
# trainer is a chainer.training.Trainer object
trainer.extend(
    chainercv.extension.DetectionVOCEvaluator(iterator, detection_model),   # Report mAP
    trigger=(1, 'epoch'))
```


<!--

## `chainer.training.Extension` for evaluation

```python
# trainer is a chainer.training.Trainer object
trainer.extend(
    chainercv.extension.DetectionVOCEvaluator(iterator, detection_model),
    trigger=(1, 'epoch'))
```

```python
evaluator = chainercv.extension.DetectionVOCEvaluator(
        iterator, detection_model)
# `result` contains dictionary of evaluation results
# ex:  result['main/map'] contains mAP
result = evaluator()
```

Internally, the evaluator runs three operations:

1. Iterate over the iterator to fetch data and make prediction.
2. Pass iterables of predictions and ground truth to `eval_*`.
3. Report results.

-->

---

template: inverse

# .large[3: Efforts on Reproducible Research]

---
<!-- THIS SLIDED IS NOT REALLY NECESSARY

## Visualization: Example Code

```python
from chainercv.datasets import VOCDetectionDataset
from chainercv.datasets import voc_detection_label_names
from chainercv.visualizations import vis_bbox
import matplotlib.pyplot as plot

*dataset = VOCDetectionDataset()
*img0, bbox0, label0 = dataset[204]
*img1, bbox1, label1 = dataset[700]

fig = plot.figure()

ax1 = fig.add_subplot(1, 2, 1)
plot.axis('off')
*vis_bbox(img0, bbox0, label0,
*        label_names=voc_detection_label_names, ax=ax1)

ax2 = fig.add_subplot(1, 2, 2)
plot.axis('off')
*vis_bbox(img1, bbox1, label1,
*        label_names=voc_detection_label_names, ax=ax2)

plot.show()

```

-->

## Bad implementations *in the wild*

- Some papers have several implementations which are publicly available
- Sometimes some of them train and evaluate the implementations with different datasets from the original papers
- It means they do not ensure that those implementations are almost identical to the original implementation which can replicate the paper results.
- And sometimes, those implementations have undocumented changes from the original implementation.

--

**.red[This is problematic for developing and comparing new ideas to the existing ones.]**

---

## ChainerCV for reproducible research

- We carefully ensured that all models provided with ChainerCV can reproduce the performances on par with the original paper results.
- When we change any part of the algorithm from the original implementation, we surely document those changes.

#### Faster R-CNN

| Training Setting | Evaluation | Reference | ChainerCV |
|:-:|:-:|:-:|:-:|
| VOC 2007 trainval | VOC 2007 test|  69.9 mAP  | **70.5 mAP** |


#### SegNet

| Training Setting | Evaluation | Reference | ChainerCV |
|:--------------:|:---------------:|:--------------:|:----------:|
| CamVid train | CamVid test | 46.3 mIoU | **47.2 mIoU**|

---

template: inverse

# .large[Comparison and Conclusions]

---

## Comparison of deep learning libraries in Computer Vision

.center[
|   | ChainerCV *  | pytorch/vision     |
|---|---|---|---|
| **Backend** | Chainer | PyTorch |
| **Supported Models** | <ul style="text-align:left"><li>Classification</li><li>Detection</li><li>Semantic segmentation</li> | <ul style="text-align:left"><li>Classification</li></ul> |
| **# of Transforms** | 17  | 11 |
| **Visualization** | .blue[Y]  | .red[N] |
| **Evaluation** | .blue[Y] | .red[N] |
]

* Combination of ChainerCV and vision related functionalities in Chainer.

This comparison is valid as of June 10th, 2017.

---

<!-- Add a demo if you want to at the first chapter

## `tfdbg`: Screencast and Demo!

.small.right[From Google Brain Team]

<div class="center">
<iframe width="672" height="378" src="https://www.youtube.com/embed/CA7fjRfduOI" frameborder="0" allowfullscreen></iframe>
</div>

<p>

.small[
<br/>
See also: [Debug TensorFlow Models with tfdbg (@Google Developers Blog)](https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html)
]
-->

---
name: last-page
class: no-number

## Concluding Remarks

We have talked about the goals of ChainerCV and some functionalities.

- It aims at providing convenient and unified interfaces to various deep learning models in computer vision.
- It also provides a set of tools to train those models on your data or public datasets.
- We try to ensure the reproducibility of the implementations with maximum effort.
- It will support researchers and engineers who try to extend conventional methods with new ideas.


.center[.middle[## Thank You!]]

<!--#### [@yuyu2172][Yusuke]-->

<!-- .footnote[Slideshow created using [remark](http://github.com/gnab/remark).] -->

<!-- vim: set ft=pandoc -->
