#!/bin/bash

HOME_DIR=/home2/xianghao
start=`date +%s`

if [ -z "$1" ]; then
    echo "navigate to $HOME_DIR/data/ ..."
    mkdir -p $HOME_DIR/data
    cd $HOME_DIR/data
else
    if [ ! -d $1 ]; then
        echo "$1 is not a valid directory"
        exit 0
    fi
    echo "navigate to $1 ..."
    cd $1
fi

echo "Downloading VOC2012 trainval ..."
# Download the data
curl -LO http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
echo "Done downloading"

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar
echo "Removing tar ..."
rm VOCtrainval_11-May-2012.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in $runtime seconds"