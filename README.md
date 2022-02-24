## Данные

<div align="center">
  <b>Dataset</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Train (scratch/clean)</b>
      </td>
      <td>
        <b>Val (scratch/clean)</b>
      </td>
      <td>
        <b>Test (scratch/clean)</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><b>White: 463 (315/148)</b></li>
            <li><b>Red: 424 (180/244)</b></li>
      </ul>
      </td>
      <td>
        <ul>
            <li><b>White: 103 (52/51)</b></li>
            <li><b>Red: 75 (36/39)</b></li>
        </ul>
      </td>
      <td>
        <ul>
            <li><b>White: 90 (53/37)</b></li>
            <li><b>Red: 77 (36/41)</b></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Пример данных  лежит в папке: <a href="https://github.com/karpovaknn/defect_detection/tree/main/dataset">dataset</a>

Полные данные можно скачать с S3 хранилища:

Для обучения использовались только изображения с деффектами. На 495 изображений получилось 1362 царапины, размер которых распределился следующим образом:
![alt text](https://github.com/karpovaknn/defect_detection/blob/main/data/train_distr.png?raw=true)

Сравнение размера царапин в тесте в разбивке по цветам с трейном:
![alt text](https://github.com/karpovaknn/defect_detection/blob/main/data/all_distr.png?raw=true)

При этом красные детали распределены равномерно по размеру царапин между train и test. 
![alt text](https://github.com/karpovaknn/defect_detection/blob/main/data/red_distr.png?raw=true)
Неравномерность распределений связана с тем, что данных с белыми деталями больше, чем с красными


## Обучение

Для обучения модели использовался <a href="https://github.com/open-mmlab/mmdetection">MMDetection</a>. 
Архитектура: <a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a> с backbone ResNeXt 101.
В качестве аугментации использовался <a href="https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/datasets/pipelines/transforms.html#PhotoMetricDistortion">PhotoMetricDistortion</a> (для изменения цвета), а также <a href="https://vfdev-5-albumentations.readthedocs.io/en/docs_pytorch_fix/_modules/albumentations/augmentations/transforms.html#ShiftScaleRotate">ShiftScaleRotate</a>.

Полный код обучения с изменением конфига в train.ipynb

Итоговый конфиг и логи обучения в <a href="https://github.com/karpovaknn/defect_detection/tree/main/job4_cascade_rcnn_x101_32x4d_fpn_1x_train_887_map0_5_0_95_aug_PMDistortion_final_fold0">job4_cascade_rcnn_x101_32x4d_fpn_1x_train_887_map0_5_0_95_aug_PMDistortion_final_fold0</a> 

Лучшие веса можно скачать тут <a href="https://wandb.ai/karpovaknn/defect_detection/artifacts/model/models_files_cascade_rcnn_x101_32x4d_fpn_1x_train_887_map0_5_0_95_aug_PMDistortion_final_fold0_job4/1815f94ab8c660b8c55e/files">epoch_16</a> 

Отчет по обучению:
![alt text](https://github.com/karpovaknn/defect_detection/blob/main/report/train.png?raw=true)
![alt text](https://github.com/karpovaknn/defect_detection/blob/main/report/val.png?raw=true)

## Результаты

|  Metric  | IOU     | Area     | MaxDets        |   All test        |   White test        |   Red test        |
| :------: | :-----: | :------: | :------------: | :----: |:------------: | :----: |  
|  Average Precision   |  0.50:0.95    |   all    |      100          | 0.302  | 0.384  | 0.099 |
|  Average Precision   |  0.50    |   all    |      100               | 0.671  | 0.830  | 0.226 |
|  Average Precision   |  0.50    |   0-40^2    |      100            | 0.059  | -1.000 | 0.059 |
|  Average Precision   |  0.50    |   40^2-70^2    |      100         | 0.404  | 0.537  | 0.417 |
|  Average Precision   |  0.50    |   70^2-96^2    |      100         | 0.851  | 0.890  | 0.374 |
|  Average Precision   |  0.50    |   96^2-1e5^2   |      100         | 0.826  | 0.830  | -1.000 |
|  Average Recall      |  0.50:0.95    |   all    |      100          | 0.371  | 0.455  | 0.165 |
|  Average Recall      |  0.50    |   all    |      100               | 0.749  | 0.890  | 0.404 |
|  Average Recall      |  0.50    |   0-40^2    |      100            | 0.059  | -1.000 | 0.059 |
|  Average Recall      |  0.50    |   40^2-70^2    |      100         | 0.659  | 0.929  | 0.533 |
|  Average Recall      |  0.50    |   70^2-96^2    |      100         | 0.896  | 0.918  | 0.667 |
|  Average Recall      |  0.50    |   96^2-1e5^2   |      100         | 0.865  | 0.865  | -1.000 |

## Начало работы

Все необходимые для работы библиотеки в requirements.txt.

Также необходимо скачать <a href="https://github.com/open-mmlab/mmdetection">MMDetection</a> с оригинального репозитория и положить рядом с defect-detection.