# Project Write-Up

This is the project for Intel® Edge AI for IoT Developers Nanodegree Program.
It's using OpenVINO toolkit for counting people.

In this project I tried 3 different models from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
- [ssd_mobilnet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
- [faster_cnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
- [ssd_mobilnet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

**Download** the model using `wget`(using *ssd_mobilnet_v2_coco* as sample):
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```
**Extract** it using `tar -xvf`(using *ssd_mobilnet_v2_coco* as sample):
```
tar -xvf sd_mobilenet_v1_coco_2018_01_28.tar.gz
```

**Convert** model to an *Intermediate Representation*(using *ssd_mobilnet_v2_coco* as sample):

```
cd ssd_mobilenet_v2_coco_2018_03_29

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

## Explaining Custom Layers
The app will not function correctly or may not able to function at all if there is unsupported layer that's not handle correctly.
The Model Optimizer will automatically classify any layer that is not in this [Supported Framework Layers List](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) as a custom layer. 
The process behind converting custom layers involves three steps, Generate, Edit, and Specify.
1) Generate the custom Layer Template Files by using the Model Extension
2) Edit the Custom Layer Template Files 
3) Specify the custom layer extension locations

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were below:

| Model | Inference Time | Size |
| ----- | -------------- | ---- |
| ssd_mobilenet_v1_coco - pre-conversion | 89 ms | 57 MB |
| ssd_mobilenet_v1_coco - post-conversion | 44 ms | 28 MB |
| faster_rcnn_inception_v2_coco - pre-conversion | 342 ms | 202 MB |
| faster_rcnn_inception_v2_coco - post-conversion | 155 ms | 98 MB |
| ssd_mobilenet_v2_coco - pre-conversion | 104 ms | 135 MB |
| ssd_mobilenet_v2_coco - post-conversion | 68 ms | 67 MB |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are
- Store capacity checker - save labor cost and safer for workers during this COV-19 situations
- Classroom attendants checker - validate the number of people to prevent fake attedants
- Home security camera - notify for unusual show up in specific time range

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as
- the room should has a normal lighting, not too bring or too dark, since this doesn't have a nightview camera. So it might fail if too dark or too bright that couldn't show the human shape.
- try to avoid width angle lens since it might got distortion on the edge. So it might not able to recongnize that's a person.