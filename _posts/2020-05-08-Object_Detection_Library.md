---
title: "Object Detection Library 소개"
date: 2020-05-08
categories: Computer Vision
---



# Object Detection Library 소개

이번 포스트는 프로젝트를 하면서 딥러닝의 한 분야인 Computer Vision 중 Object detection 과제를 수행하면서 사용한 라이브러리를 소개합니다.



## Object Detection이란

![Computer Vision task](/assets/postimgs/fig1_cv_task.png)*그림: 대표적인 Computer Vision task*

Object Detection을 알기 전에 Image Classification에 대해 알아 보도록 합니다. DNN에 입력으로 이미지를 넣으면 그 이미지에 해당하는 Class를 분류해내는 문제를 Image Classification 이라 부르며, 위 그림과 같이 타겟으로 하는 전체 class에 대한 확률 값들을 출력하게 됩니다. 

**Object Detection은 Image Classification task에 사물의 위치를 Bounding Box로 예측하는 Regression task가 추가된 문제**입니다. 

일반적으로 Object Detection 이라 부르는 문제는 한 이미지에 여러 class의 객체가 동시에 존재할 수 있는 상황을 가정합니다. 즉, **multi-labeled classification** (한 이미지에 여러 class 존재)과 **bounding box regression** (box의 좌표 값을 예측) 두 문제가 합쳐져 있다고 생각하면 됩니다. 위 그림과 같이 하나의 이미지에 여러 객체가 존재하여도 검출이 가능하여야 합니다. [^1]

> Object Detection = Multi-labeled Classification + Bounding Box Regression



## Faster R-CNN

> Faster R-CNN(Faster Region-based Convolutional Neural Network)은 객체 탐지 과업 을 수행하는 딥러닝 기반 모델

![Faster R-CNN Network](/assets/postimgs/Faster-R-CNN.JPG)

*그림: Faster R-CNN Network*

**Faster R-CNN**은 중심에 위치한 **Convolution Neural Network (CNN)**와 이 CNN 의 output인 final feature map을 공유하는 두 개의 sub-network로 구성됩니다. 두 sub-network 는 각각 **Region Proposal Network (RPN)**과 **Object Detector**에 해당하며, 전자는 입력 이미지에 대해 후보 region을 제안하는 네트워크이고, 후자는 제안된 region들 각각에 대해 어떤 object인지 탐지하는 네트워크입니다.[^2]

Faster R-CNN 알고리즘에 더 자세한 내용은 기회가 된다면 추후에 자세히 다뤄보도록 하겠습니다. 



Object Detection 문제를 해결하기 위해 다음과 같은 라이브러리를 사용하였습니다.

- Detecto
- Detectron2

이번 포스트에서는 두 라이브러리에 대해 간략하게 소개하며 좀 더 자세한 사용 방법에 대해서는 다음 포스트를 통해 작성할 예정입니다.



## 1. Detecto

> PyTorch 기반 Faster R-CNN, ResNet-50 FPN 네트워크를 기반으로 학습된 모델을 활용할 수 있는 파이썬 패키지

[![detecto API logo](https://github.com/alankbi/detecto/raw/master/assets/logo_words.svg?sanitize=true)](https://github.com/alankbi/detecto)

*그림: Detecto logo(이미지를 누르면 해당 패키지의 github 페이지로 이동합니다.*



**Detecto 사용 예시 이미지**

| Still Image                                                  | Video                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Detecto still image](https://github.com/alankbi/detecto/raw/master/assets/apple_orange.png)](https://github.com/alankbi/detecto/blob/master/assets/apple_orange.png) | [![Video demo of Detecto](https://github.com/alankbi/detecto/raw/master/assets/demo.gif)](https://github.com/alankbi/detecto/blob/master/assets/demo.gif) |



#### **Detecto API 특징**

- **API 사용을 위한 환경 설정이 간편함**

  pip을 사용하여 Detecto를 설치할 경우, 다음과 같은 명령어를 입력하여 실행하면 됩니다.
  ```pip install detecto ```
  pip를 사용하여 설치하면 Detecto API를 사용하기 위해 요구되는 패키지들이 자동으로 설치됩니다. 만약 설치가 제대로 되지 않을 경우 관련 요구환경을 직접 입력하여 다운받도록 합니다. (requirements list: [requirements.txt](https://github.com/alankbi/detecto/blob/master/requirements.txt))

- **VOC Format 제공**

- **간결한 코드로 객체 인식 모델 생성 가능**
  Detecto API는 간단한 코드 사용만으로 이미 학습된 모델 사용과 모델 학습을 시킬 수 있다는 점이 가장 큰 장점입니다.

  - Model Zoo를 불러와서 모델을 실행시키기 위한 코드는 다음과 같습니다.

  ```python
  from detecto.core import Model
  from detecto.visualize import detect_video
  
  model = Model()  # Initialize a pre-trained model
  detect_video(model, 'input_video.mp4', 'output.avi')  # Run inference on a video
  ```

  - Custom Dataset을 사용해 모델을 학습시키기 위한 코드는 다음과 같습니다.

  ```python
  from detecto.core import Model, Dataset
  
  dataset = Dataset('custom_dataset/')  # Load images and label data from the custom_dataset/ folder
  
  model = Model(['dog', 'cat', 'rabbit'])  # Train to predict dogs, cats, and rabbits
  model.fit(dataset)
  
  model.predict(...)  # Start using your trained model!
  ```


  하지만 개인적으로 사용해본 결과, 다음과 같이 사용하면 학습 과정 중 오류가 났을 경우, 문제 원인 파악이 어려우며 loss와 같은 지표 확인이 어렵기 때문에 추가적인 코드 작성이 필요합니다. 이는 다음 포스트를 통해 소개하도록 하겠습니다.

  

  **Detecto API는 딥러닝과 Object Detection 모델을 처음 사용하는 입문자들에게 추천**합니다. 
  Object Detection 문제를 해결하기 위한 일련의 과정들을 빠르고 쉽게 경험해 볼 수 있어서 해당 라이브러리를 사용한 뒤에 다른 라이브러리를 사용하면 좀 더 이해하기 쉽습니다.



## 2. Detectron2

> Facebook AI Research에서 개발한 PyTorch기반의 Object Detection, segmentation API

[![Detectron2 logo](https://github.com/facebookresearch/detectron2/raw/master/.github/Detectron2-Logo-Horz.svg?sanitize=true)](https://github.com/facebookresearch/detectron2)

*그림: Detectron2 logo(이미지를 누르면 해당 라이브러리의 github 페이지로 이동합니다.*



**Detectron2 사용 예시 이미지**

![img](https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png)



#### Detectron2 특징

- **다양한 Object Detection 알고리즘 제공**
  Detectron2 는 object detection 뿐만 아니라 semantic segmentation, panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes 등의 다양한 기능들을 제공합니다.

- **방대한 Pre-trained Model Zoo 제공**
  Model zoo를 사용하기 위해선 공통 모델 아키텍처를 생성한 뒤에 Detectron2에서 제공하는 다양한 사전 훈련된 weights를 선택하여 지정해주기만 하면 됩니다.
  사용가능한 baseline들은 [MODEL_ZOO.md](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) 에서 확인할 수 있습니다.

- **모델 학습 시간이 빠름**
  ![training throughput of R50-FPN Maks R-CNN](/assets/postimgs/image-20200508154827433.png)

  위의 이미지는 R50-FPN 기반의 Mask R-CNN을 학습시켰을 경우 Detectron2의 학습 시간이 제일 빠르다고 소개하고 있습니다. 필자가 사용해본 결과 위에서 소개한 Detecto와 비교했을 때도 학습속도가 매우 빠른 것을 확인할 수 있었습니다.



#### **Detectron2 설치 방법**

사실 이 부분에 대해 설명하고 싶지만, Detectron2를 사용하기 위해서 가장 우선시 되는 환경이 Linux와 MacOS라 단계별로 실행하면서 설명해드리기가 어렵네요. (~~필자는 windows10을 사용하고 있습니다T.T~~) 기회가 된다면 Dockerfile을 통해 설치 및 사용과정에 대해 소개해보도록 하겠습니다.

자세한 설치방법 및 요구환경은 [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) 를 통해 확인할 수 있으며, 설치과정 없이 [Google Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=QHnVupBBn9eR)으로 제공되는 튜토리얼을 통해 Detectron2를 사용해볼 수 있습니다.



Detectron2는 Detecto에 비해 사용방법이 다소 어렵게 느껴질 수도 있습니다. 하지만 튜토리얼을 통해 사용방법을 익힌다면 그다지 어렵지 않다는 걸 알 수 있으실 겁니다. 특히 [Mask R-CNN](https://github.com/matterport/Mask_RCNN) 을 사용해보신 분들이라면 사용법이 매우 유사하기 때문에 금방 사용해 보실 수 있습니다.

제가 프로젝트를 수행하면서 Custom Dataset에 맞게 사용한 방법은 다음 포스트를 통해 소개해보도록 하겠습니다.





## Reference


  [^1]: https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-what-is-object-detection/

- Faster R-CNN 

  - 그림 출처: https://arxiv.org/pdf/1506.01497.pdf


    [^2]: 고광은, 심귀보. (2017). 딥러닝을 이용한 객체 인식 및 검출 기술 동향. 제어로봇시스템학회지, 23(3), 17-24.

- Detecto API

  - https://detecto.readthedocs.io/en/latest/api/
  - https://github.com/alankbi/detecto

- Detectron2

  - https://github.com/facebookresearch/detectron2

