# Guns-Dataset
The repository contains labelled images of guns taken from various sources.
<br>

* Run below command inside create_gun_tfrecord
```sh
git clone https://github.com/tensorflow/models.git
```

* Run protoc on the object detection repo
```sh
cd models/research && protoc object_detection/protos/*.proto --python_out=.
```