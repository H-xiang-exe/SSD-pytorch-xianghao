import torch
import torch.nn as nn


class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super(BoxPredictor, self).__init__()
        self.cfg = cfg
        self.loc_reg_headers = nn.ModuleList()
        self.box_cls_headers = nn.ModuleList()
        for feature_level, (boxes_per_location, feature_output_channels) in enumerate(
                zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUTPUT_CHANNELS)):
            self.loc_reg_headers.append(self.loc_reg_block(feature_level, feature_output_channels, boxes_per_location))
            self.box_cls_headers.append(self.box_cls_block(feature_level, feature_output_channels, boxes_per_location))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loc_reg_block(self, level, input_channels, boxes_per_location):
        raise NotImplementedError

    def box_cls_block(self, level, input_channels, boxes_per_location):
        raise NotImplementedError

    def forward(self, features):
        """

        Args:
            features: [(N, 512, 38, 38), (N, 1024, 19, 19), (N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3),
                   (N, 256, 1, 1)]

        Returns:

        """
        locs_pred = []
        confs_pred = []
        for feature, loc_layer, cls_layer in zip(features, self.loc_reg_headers, self.box_cls_headers):
            locs_pred.append(loc_layer(feature).permute(0, 2, 3, 1).contiguous())
            confs_pred.append(cls_layer(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        locs_pred = torch.cat([loc.view(batch_size, -1, 4) for loc in locs_pred], dim=1)
        confs_pred = torch.cat([conf.view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES) for conf in confs_pred], dim=1)

        return locs_pred, confs_pred


class SSDBoxPredictor(BoxPredictor):
    def loc_reg_block(self, level, input_channels, boxes_per_location):
        out_channels = boxes_per_location * 4
        return nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1)

    def box_cls_block(self, level, input_channels, boxes_per_location):
        out_channels = boxes_per_location * self.cfg.MODEL.NUM_CLASSES
        return nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1)


def make_box_predictor(cfg):
    return SSDBoxPredictor(cfg)
