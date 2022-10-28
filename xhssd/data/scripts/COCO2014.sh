#!/bin/bash
#第1行指明该脚本用什么shell解释，否则使用默认shell
#'#'表示注释当前行

start=`date +%s`

HOME_DIR=/home/xianghao
if [ -z "$1" ]; then
  # navigate to HOME_DIR/data
  echo "navigating to $HOME_DIR/data/ ..."
  mkdir -p $HOME_DIR/data
  cd $HOME_DIR/data
  mkdir -p ./coco
  cd ./coco
  mkdir -p ./images
  mkdir -p ./annotations
else
  # check if is valid directory
  if [ ! -d $1 ]; then
    echo "$1 is not a valid directory"
    exit 0
  fi
  echo "navigating to $1 ..."
  cd $1
fi

if [ ! -d images ]; then
  mkdir -p ./images
fi

# Download the image data.
cd ./images
echo "Downloading MSCOCO train images ..."
curl -LO http://images.cocodataset.org/zips/train2014.zip
echo "Downloading MSCOCO val images ..."
curl -LO http://images.cocodataset.org/zips/val2014.zip

cd ../
if [ ! -d annotations ]; then
  mkdir -p ./annotations
fi

# Download the annotation data.
cd ./annotations
echo "Downloading MSCOCO train/val annotations ..."
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2014.zip
echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting train images ..."
unzip ../images/train2014.zip -d ../images
echo "Extracting val images ..."
unzip ../images/val2014.zip -d ../images
echo "Extracting annotations ..."
unzip ./annotations_trainval2014.zip

echo "Removing zip files ..."
rm ../images/train2014.zip
rm ../images/val2014.zip
rm ./annotations_trainval2014.zip

echo "Creating trainval35k dataset ..."
curl -LO https://s3.amazonaws.com/amdegroot-datasets/instances_trainval35k.json.zip

# combine train and val
echo "Combining train and val images"
mkdir ../images/trainval35k
cd ../images/train2014
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} + # dir too large for cp
cd ../val2014
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} +

end=`date +%s`
runtime=$((end-start))

echo "Completed in $runtime seconds"

