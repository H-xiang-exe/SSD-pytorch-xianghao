import os

import argparse

from .coco import COCODetection

VOC_ROOT="/home2/xianghao/data/VOCdevkit/"


parser = argparse.ArgumentParser(description="Single Shot MultiBox Detector Training With Pytorch")
parser.add_argument("--dataset", default='VOC', choices=["VOC", "COCO"], type=str, help="VOC or COCO")
parser.add_argument("--dataset_root", default=VOC_ROOT, help="Dataset root directory path")

args = parser.parse_args()

def train():
    if args.dataset == "COCO":
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error("Must specify dataset_root if specifying dataset")
            print("WARGING: Using default COCO dataset_root because --dataset_root was not specified")
            args.dataset_root=COCO_ROOT
        cfg = coco
        dataset = COCODetection()
    elif args.dataset == "VOC":
        if args.dataset_root == COCO_ROOT:
            parser.error("Must specify dataset_root if specifying dataset_root")
        cfg = voc
        dataset = VOCDetection()
        

if __name__ == "__main__":
    train()