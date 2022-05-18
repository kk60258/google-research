#!/usr/bin/env bash

echo "Please make sure current path is under seethroughkeyboard/"
WORKSPACE=`pwd`
export WORKSPACE

NAME=$1

if [ -z $NAME ]; then
    echo "Please provide arguments. 1 for install tensorflow cpu; 2 for install tensorflow gpu"
    exit 0
fi

case $NAME in
  1*)
    ##install tensorflow cpu
    pip install tensorflow==1.12.0
    ;;
  2*)
    ##install tensorflow gpu
    pip install tensorflow-gpu==1.12.0
    ;;
  *)
    echo "Cannot find 1 or 2: ${NAME}."
    ####exit 1
    ;;
esac

sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython
pip install contextlib2
pip install matplotlib
pip install absl-py
pip install lxml
pip install Pillow

### manual install protobuf-compiler
cd $WORKSPACE
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

# From google_obj_detection/
./bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo 'export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim' >> ~/.bashrc 

python object_detection/builders/model_builder_test.py

##ImportError: No module named 'pycocotools'
#### install coco api
cd $WORKSPACE
cd ..
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
make install
#python setup.py install


