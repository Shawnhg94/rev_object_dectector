import os
import numpy as np
import torch
from PIL import Image
from label_manager import LabelManager
from label_manager import parse_label
import pandas as pd
import json


class RevDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, input_df = None, target_df = None):
        self.root = root
        self.transforms = transforms

        if input_df is not None:
            self.input_df = input_df
        else:
            self.input_df = pd.read_csv('input.csv')
        if target_df is not None:
            self.target_df = target_df
        else:
            self.target_df = pd.read_csv('target.csv')


        self.log = True
        self.label_manager = LabelManager()
        
    def __getitem__(self, idx):
        row_data = self.input_df.iloc[idx].to_dict()
        path = row_data['path']
        w = int(row_data['width'])
        h = int(row_data['height'])
        channels = int(row_data['channels'])
        #print('path', path, 'w', w, 'h', h, 'channels', channels)
        pixels_str = row_data['pixels']
        pixel_values = np.array(list(map(int, pixels_str.split(','))))
        if channels == 1:
            mode = 'L'  # Grayscale
            expected_shape = (h, w)
        elif channels == 3:
            mode = 'RGB'
            expected_shape = (h, w, 3)

        img_array = pixel_values.reshape(expected_shape).astype(np.uint8)
        img = Image.fromarray(img_array, mode=mode)

        #Target
        row_data = self.target_df.iloc[idx].to_dict()
        path = row_data['path']
        w = int(row_data['width'])
        h = int(row_data['height'])
        channels = int(row_data['channels'])
        #print('path', path, 'w', w, 'h', h, 'channels', channels)
        pixels_str = row_data['pixels']
        pixel_values = np.array(list(map(int, pixels_str.split(','))))
        if channels == 1:
            mode = 'L'  # Grayscale
            expected_shape = (h, w)
        elif channels == 3:
            mode = 'RGB'
            expected_shape = (h, w, 3)
        elif channels == 4:
            mode = 'RGBA'
            expected_shape = (h, w, 4)
        img_array = pixel_values.reshape(expected_shape).astype(np.uint8)
        mask_img = Image.fromarray(img_array, mode=mode)

        label_map = row_data['object_label_map']
        target_label = json.loads(label_map)

        # convert the PIL Image into a numpy array
        mask = np.array(mask_img)

        if (self.log) :
            print('mask shape', mask.shape)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]


        masks = mask == obj_ids[:, None, None]
        if (self.log) :
            print('masks shape', mask.shape)

        # get bounding box coordinates for each mask
        #print('idx', idx)
        num_objs = len(obj_ids)
        if (self.log) :
            print('num_objs: ', num_objs, 'obj_ids', obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                raise AttributeError
                continue
            xmin_val = np.min(pos[1])
            xmax_val = np.max(pos[1])
            ymin_val = np.min(pos[0])
            ymax_val = np.max(pos[0])

            if xmin_val >= xmax_val:
                if xmax_val < masks.shape[2] - 1: # If there's space to expand to the right
                    xmax_val = xmin_val + 1
                elif xmin_val > 0: # If at the right edge, expand to the left
                    xmin_val = xmax_val - 1
                else: 
                    xmax_val = xmin_val + 0.1 # Ensure float and slightly larger

            # Ensure ymax is strictly greater than ymin
            if ymin_val >= ymax_val:
                if ymax_val < masks.shape[1] - 1: # If there's space to expand down
                    ymax_val = ymin_val + 1
                elif ymin_val > 0: # If at the bottom edge, expand up
                    ymin_val = ymax_val - 1
                else: # Similar to width, for 1-pixel height
                    ymax_val = ymin_val + 0.1

            # Convert to float, as downstream transforms and model expect float boxes
            current_box = [float(xmin_val), float(ymin_val), float(xmax_val), float(ymax_val)]

            # Final check, this should ideally not be needed if above logic is perfect for all edge cases
            if not (current_box[2] > current_box[0] and current_box[3] > current_box[1]):
                if not (current_box[2] > current_box[0]):
                    current_box[2] = current_box[0] + 0.1
                if not (current_box[3] > current_box[1]):
                    current_box[3] = current_box[1] + 0.1
                if (self.log):
                    print(f"Warning: Adjusted a degenerate box for image_id {idx}, obj_id {obj_ids[i]}: to {current_box}")


            boxes.append(current_box)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels =[]
        for i in obj_ids:
            if (self.log):
                print('obj', obj_ids, 'target', target_label)
            label = target_label[str(i)]
            labels.append(label)


        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.input_df)
    