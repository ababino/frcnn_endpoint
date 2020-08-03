import os
import logging

from flask import Flask
from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from PIL import Image
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torchvision
import torch


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

@api.route('/')
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
        logger.info('{} objects detected'.format(len(coco['annotations'])))

        return {'coco': coco}

