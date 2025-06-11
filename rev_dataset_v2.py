import os
import numpy as np
import torch
from PIL import Image
from label_manager import LabelManager
from label_manager import parse_label


class RevDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.imgs = list(sorted(os.listdir('rev_input/')))
        self.masks = sorted([f for f in os.listdir('rev_mask/') if f.endswith('.png')])
        self.lables = sorted([f for f in os.listdir('rev_mask/') if f.endswith('.yaml')])
        self.log = False
        self.label_manager = LabelManager()

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img_path = os.path.join('rev_input/', self.imgs[idx])
        mask_path = os.path.join('rev_mask/', self.masks[idx])
        label_path = os.path.join('rev_mask/', self.lables[idx])

        # img = Image.open(img_path).convert("RGB")
        img = Image.open(img_path)
        # img = Image.open(img_path).convert("L")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)

        if (self.log) :
            print('mask shape', mask.shape, 'img_path: ', img_path)

        # label_num = self.label_manager.get_num()
        # mask_label = np.zeros(label_num, (mask.shape[0], mask.shape[1]), np.uint8)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]


        # split the color-encoded mask into a set
        # of binary masks

        # for i in obj_ids:
        #     mask_label[i] = mask == obj_ids[i, None, None]

        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        #print('idx', idx)
        num_objs = len(obj_ids)
        if (self.log) :
            print('num_objs: ', num_objs, 'obj_ids', obj_ids, 'img_path: ', img_path)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        target_label = parse_label(label_path)
        
        labels =[]
        for i in obj_ids:
            if (self.log):
                print('obj', obj_ids, 'target', target_label)
            label = target_label[i]
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

        if (self.log) :
            print('img_path: ', img_path, 'boxes:', boxes.shape, 'labels:', labels.shape, 'masks:', masks.shape, 'image_id:', image_id.shape, 'area:', area.shape, 'iscrowd:', iscrowd.shape, 'num_objs', num_objs)
            # print('boxes:', boxes.shape)
            # print('labels:', labels.shape)
            # print('masks:', masks.shape)
            # print('image_id:', image_id.shape)
            # print('area:', area.shape)
            # print('iscrowd:', iscrowd.shape)
            self.log = False

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)