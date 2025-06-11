from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import warnings

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from label_manager import LabelManager

warnings.filterwarnings('ignore')


class ImageSegmentation:
    
    def __init__(self):
        # from object.yaml
        self.num_classes = 29
        self.model = self.build_default_model(self.num_classes)
        self.model.load_state_dict(torch.load('mask-rcnn.pt', weights_only=True))
        self.model.eval()
    

    def build_default_model(self, num_classes):
      print('build_model - num_classes: ', num_classes)
      # load an instance segmentation model pre-trained on COCO
      model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pre_trained = False)

      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

      # Stop here if you are fine-tunning Faster-RCNN

      # now get the number of input features for the mask classifier
      in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
      hidden_layer = 256
      # and replace the mask predictor with a new one
      model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
      return model
    
    def get_prediction(self, img_path, confidence):  
      label_manager = LabelManager()      
      CLASS_NAMES = ['__background__'] + label_manager.get_names()
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      self.model.to(device)

      img = Image.open(img_path)
      # img = Image.open(img_path).convert("L")
      transform = T.Compose([T.ToTensor()])
      img = transform(img)
      img = img.to(device)
      pred = self.model([img])
      pred_score = list(pred[0]['scores'].detach().cpu().numpy())
      # print('pred_score', pred_score)
      pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
      masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
      # print("pred[0]['masks']", pred[0]['masks'])
      
      
      # print(pred[0]['labels'].numpy().max())
      pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
      pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
      masks = masks[:pred_t+1]
      # print('masks', masks.shape)
      pred_boxes = pred_boxes[:pred_t+1]
      # print('pred_boxes', pred_boxes.shape)
      pred_class = pred_class[:pred_t+1]
      labels = pred[0]['labels'].cpu().numpy()
      labels = labels[:pred_t+1]
      # print('labels', labels)
      return masks, pred_boxes, pred_class, labels
    
    def get_coloured_mask(self, mask, colour):
      r = np.zeros_like(mask).astype(np.uint8)
      g = np.zeros_like(mask).astype(np.uint8)
      b = np.zeros_like(mask).astype(np.uint8)
      r[mask == 1], g[mask == 1], b[mask == 1] = colour#[66, 135, 245] #colours[random.randrange(0,10)]
      coloured_mask = np.stack([r, g, b], axis=2)
      return coloured_mask

    def segment_instance(self, img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
        label_manager = LabelManager()
        masks, boxes, pred_cls, labels = self.get_prediction(img_path, confidence)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
          label = labels[i]
          colour = label_manager.get_colour(label)
          
          rgb_mask = self.get_coloured_mask(masks[i], colour)
          img = cv2.addWeighted(img, 0.8, rgb_mask, 0.3, 0)
          pt1 = tuple([int(j) for j in boxes[i][0]])
          pt2 = tuple([int(j) for j in boxes[i][1]])
        #   cv2.rectangle(img, pt1, pt2,color=(0, 255, 0), thickness=rect_th)
        #   cv2.putText(img,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    

if __name__ == '__main__':
    image_segment = ImageSegmentation()
    image_segment.segment_instance('input/2.jpg', confidence=0.7)