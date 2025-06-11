from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
import yaml


class ObjectEntity:
    def __init__(self, name: str, id: int, colour: list):
        self.name = name
        self.id = id
        self.colour = colour

    def __str__(self):
        return "Name: {}, ID: {}, Colour: {}".format(self.name, self.id, self.colour)

class LabelManager:
    def __init__(self):
        with open("object.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        self.obj_entity_map = {}
        self.class_names = []
        for ele in config['DRIVING_objects']:
            name = ele['Entity']
            id = ele['ID']
            colour = ele['Colour']
            entity = ObjectEntity(name, id=id, colour=colour)
            self.obj_entity_map.update({id: entity})
            self.class_names.append(name)
        
    
    def get_colour(self, id:int):
        return self.obj_entity_map[id].colour
    

    def get_num(self):
        return len(list(self.obj_entity_map.keys()))
    
    def get_ids(self):
        return list(self.obj_entity_map.keys())
    
    def get_names(self):
        return self.class_names
    
    def get_name(self, id:int):
        return self.class_names[id]
    
    def get_label(self, name:str):
        index = self.class_names.index(name)
        return index + 1



class RevVision:
    def __init__(self):
        self.label_manager = LabelManager()
        self.num_classes = self.label_manager.get_num() + 1
        self.model = self.build_default_model(self.num_classes)
        # self.model.load_state_dict(torch.load('rev_model.pt', weights_only=True))
        self.model.load_state_dict(torch.load('rev_demo.pt', weights_only=True))
        self.model.eval()

        self.class_names = ['__background__'] + self.label_manager.get_names()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        # Transform
        self.transform = T.Compose([T.ToTensor()])
        self.entities = [1, 2, 3, 16, 17]

    
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



    def get_prediction(self, img, confidence):
        img = self.transform(img)

        img = img.to(self.device)
        pred = self.model([img])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_list = [pred_score.index(x) for x in pred_score if x>confidence]
        if (len(pred_list) == 0):
            return [], [], [], []
        pred_t = pred_list[-1]
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()

        pred_class = [self.class_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        labels = pred[0]['labels'].cpu().numpy()
        labels = labels[:pred_t+1]

        return masks, pred_boxes, pred_class, labels
    
    def segment_instance(self, image, frame_id, confidence=0.7, rect_th=1, text_size=0.5, text_th=1, label_filter = 0, capture = False):
        start_time = time.perf_counter()
        masks, boxes, pred_cls, labels = self.get_prediction(image, confidence)
        end_time = time.perf_counter()
        print('detecting labels: ', labels, 'processing_time: ', end_time-start_time)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h = image.height
        w = image.width
        out_img = np.zeros((h, w, 4), np.uint8)
        img_array = np.array(image)
        capture_array = np.array(image).copy()
        #print('img_array', img_array.shape)
        
        for i in range(len(masks)):
            # rgb_mask = get_coloured_mask(masks[i])
            # img = cv2.addWeighted(img, 0.8, rgb_mask, 0.3, 0)
            label = labels[i]
            if (label_filter > 0 and label_filter != label):
                continue
            colour = self.label_manager.get_colour(label)
            # Efficiently apply color and alpha to the mask using NumPy
            mask = masks[i] > 0  # Create a boolean mask where the condition is true
            
            if capture:
                masked_pixels = capture_array[mask]

                # Get the bounding box to create a white background image
                mask_row_indices, mask_col_indices = np.where(mask)
                min_row, max_row = np.min(mask_row_indices), np.max(mask_row_indices)
                min_col, max_col = np.min(mask_col_indices), np.max(mask_col_indices)
                masked_height = max_row - min_row + 1
                masked_width = max_col - min_col + 1

                # Create a white background image (RGB)
                white_background = np.full((masked_height, masked_width, 3), 255, dtype=np.uint8)

                # Get the corresponding pixels from the original image within the mask's bounding box
                cropped_masked_pixels = capture_array[min_row:max_row + 1, min_col:max_col + 1]

                # Create a mask for the cropped region
                cropped_mask = mask[min_row:max_row + 1, min_col:max_col + 1]

                # Apply the masked pixels onto the white background
                white_background[cropped_mask] = cropped_masked_pixels[cropped_mask]

                masked_region_image = Image.fromarray(white_background)
                masked_region_image.save(f'capture/{pred_cls[i]}_{frame_id}_{i}.png')

                alpha_channel = np.full((masked_pixels.shape[0], 1), 255, dtype=np.uint8)
                colored_mask = np.concatenate((masked_pixels, alpha_channel), axis=1)
                # out_img[mask] = colored_mask_rgba
                
            else:
                colored_mask = np.array(colour + [153], dtype=np.uint8)
            out_img[mask] = colored_mask
            # for y in range(0, h):
            #     for x in range(0, w):
            #         if (masks[i][y, x] > 0):
            #             # alpha 60% = 153
            #             out_img[y, x] = colour + [153]
            if (label in self.entities):
                pt1 = tuple([int(j) for j in boxes[i][0]])
                pt2 = tuple([int(j) for j in boxes[i][1]])
                #print('pt1: ', pt1, 'pt2: ', pt2)
                cv2.rectangle(img_array, pt1, pt2,color=colour, thickness=rect_th)
                cv2.putText(img_array,pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, colour,thickness=text_th)

        return Image.fromarray(img_array), Image.fromarray(out_img, mode='RGBA'), pred_cls, labels