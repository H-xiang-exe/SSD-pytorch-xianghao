#!/bin/bash

HOME_DIR=/home2/xianghao
start=`date +%s`

# handle optional download
if [ -z "$1" ]; then
    # navigate to HOME_DIR/data/ ...
    echo "navigate to $HOME_DIR/data/ ..."
    mkdir -p $HOME_DIR/data
    cd $HOME_DIR/data
else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo "$1 is not a valid directory"
        exit 0
    fi
    echo "navigate to $1 ..."
    cd $1
fi

echo "Downloading VOC2007 trainval ..."
# Download the data
curl -LO http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
echo "Downloading VOC2007 test data ..."
curl -LO http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
echo "Done downloading"

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar
echo "removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in $runtime seconds"