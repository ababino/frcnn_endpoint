import os
import logging
import time

from flask import Flask
from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from PIL import Image
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torchvision
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import numpy as np

logger = logging.getLogger('gunicorn.error')

api = Namespace('frcnn', description='Model related operations')
image_upload = reqparse.RequestParser()
image_upload.add_argument('image', location='files', type=FileStorage, required=True, help='Image')

categories = os.getenv('CATEGORIES').split(',')
device = int(os.getenv('GPU_DEVICE'))

logger.info('Loading model')
backbone = resnet_fpn_backbone(os.getenv('BACKBONE'), False)
model = FasterRCNN(backbone, len(categories))
state_dict = torch.load(os.getenv('MODEL_PATH'))
model.load_state_dict(state_dict['model'])
model.to(device);
model.eval();
logger.info('Model ready')

logger.info('Loading detecton model')
cfg = get_cfg()
cfg.merge_from_file(os.getenv('DETECTRON_CONFIG'))
cfg.MODEL.WEIGHTS = os.getenv('DETECTRON_MODEL') 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.freeze()
predictor = DefaultPredictor(cfg)
logger.info('Detecton model ready')

@api.route('/vision')
class FRCNN(Resource):

    @api.expect(image_upload)
    def post(self):
        """ send image get COCO """
        args = image_upload.parse_args()
        im = args.get('image')

        im = Image.open(im).convert('RGB')
        width, height = im.size
        im = torchvision.transforms.ToTensor()(im)
        im = im.to(device)
        det = model([im])[0]

        coco = {'images': [{'width': width, 'height': height}], "categories": [], "annotations": []}
        boxes = det['boxes']
        for i, roi in enumerate(boxes):
            bbox = [roi[0], roi[1], roi[0], roi[3], roi[2], roi[3], roi[2], roi[1]]
            bbox = [int(x) for x in bbox]
            cat_id = int(det['labels'][i])
            coco['categories'].append({"id": cat_id, "name": categories[cat_id]})
            coco['annotations'].append({"segmentation": [bbox], 'category_id': cat_id, 'isbbox': True})
        logger.info('{} instances detected'.format(len(coco['annotations'])))

        return {'coco': coco}

@api.route('/detectron')
class Detectron_FRCNN(Resource):

    @api.expect(image_upload)
    def post(self):
        """ send image get COCO """
        args = image_upload.parse_args()
        im = args.get('image')

        im = Image.open(im).convert('RGB')
        im = np.asarray(im)
        im = im[:, :, ::-1]
        t0 = time.time()
        outputs = predictor(im)
        instances = outputs['instances'].to(torch.device('cpu'))

        coco = {'images': [{'width': im.shape[1], 'height': im.shape[0]}], "categories": [], "annotations": []}
        boxes = instances.pred_boxes
        pred_classes = instances.pred_classes
        scores = instances.scores
        for roi, pred_class, score in zip(boxes, pred_classes, scores):
            roi = [int(x) for x in roi]
            bbox = [roi[0], roi[1], roi[0], roi[3], roi[2], roi[3], roi[2], roi[1]]
            cat_id = int(pred_class)+1
            coco['categories'].append({"id": cat_id, "name": categories[cat_id]})
            coco['annotations'].append({"segmentation": [bbox],
                                        'category_id': cat_id,
                                        'isbbox': True,
                                        'score': float(score),
                                        'bbox': roi})
        inference_time = time.time()-t0
        logger.info('{} instances detected in {}s'.format(len(coco['annotations']), inference_time))
        return {'coco': coco}
