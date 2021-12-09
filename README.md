# Raspberry_Camera
Tensorflow on Jetson Nano, OpenCV, SSDmobile

## 1.Download NUMPY ONNX PYCUDA and Install them
<pre><code># Install TF1.8.0
python3 -m pip install onnx==1.4.1
python3 -m pip install numpy==1.16.2
python3 -m pip install pycuda==2018.4

# Get model
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz .
</code></pre>
[Link: TensorFlow on arm](https://github.com/lhelontra/tensorflow-on-arm/releases),
[Link: SSD mobile](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz),
[Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
<pre><code>
git clone https://github.com/AlexeyAB/darknet.git
</code></pre>

## 2.Check OpenCV for python
<pre><code>python3 -m pip install opencv==4.4.1
</code></pre>

## 3.weight2onnx
<pre><code>python3 yolo_to_onnx.py -m yolov4_tiny</code></pre>

## 4.onnx2tensorRT
<pre><code>python3 onnx_to_tensorrt.py -m yolov4_tiny</code></pre>

## 5.Test
By default, you will open the camera, display the images captured.

And run the Object Detection Demo.
<pre><code>python opencv_camera.py</code></pre>

Have Fun~
