# data_utils.py
import os
import re
import cv2
import numpy as np
import selectivesearch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset

def load_penn_fudan_data(root_dir):
    """
    Penn-Fudan Pedestrian 데이터셋에서 Annotation 폴더 내 텍스트 파일을 읽어,
    이미지 파일 경로와 bounding box 정보를 추출하는 함수.
    
    각 텍스트 파일 예시:
        # Compatible with PASCAL Annotation Version 1.00
        Image filename : "PennFudanPed/PNGImages/FudanPed00001.png"
        Image size (X x Y x C) : 559 x 536 x 3
        Database : "The Penn-Fudan-Pedestrian Database"
        Objects with ground truth : 2 { "PASpersonWalking" "PASpersonWalking" }
        ...
        # Details for pedestrian 1 ("PASpersonWalking")
        Original label for object 1 "PASpersonWalking" : "PennFudanPed"
        Bounding box for object 1 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)
        Pixel mask for object 1 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00001_mask.png"
        ...
        # Details for pedestrian 2 ("PASpersonWalking")
        Original label for object 2 "PASpersonWalking" : "PennFudanPed"
        Bounding box for object 2 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (420, 171) - (535, 486)
        Pixel mask for object 2 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00001_mask.png"
    
    반환:
        data_list = [
            {
                'img_path': 'PennFudanPed/PNGImages/FudanPed00001.png',
                'boxes': [[160, 182, 302, 431], [420, 171, 535, 486]],
                'labels': [1, 1]
            },
            ...
        ]
    """
    annotation_dir = os.path.join(root_dir, "Annotation")
    data_list = []
    
    # Annotation 폴더 내의 모든 텍스트 파일들을 찾음
    txt_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.txt')])
    
    for txt_file in txt_files:
        txt_path = os.path.join(annotation_dir, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        img_path = None
        boxes = []
        labels = []
        
        for line in lines:
            line = line.strip()
            # 이미지 파일 경로 추출 (예: Image filename : "PennFudanPed/PNGImages/FudanPed00001.png")
            if line.startswith("Image filename"):
                m = re.search(r'Image filename\s*:\s*"([^"]+)"', line)
                if m:
                    img_path = m.group(1)
            # bounding box 정보 추출
            # 예: Bounding box for object 1 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)
            if line.startswith("Bounding box for object"):
                m = re.search(
                    r'Bounding box for object \d+ ".*?" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+),\s*(\d+)\) - \((\d+),\s*(\d+)\)',
                    line
                )
                if m:
                    x_min = int(m.group(1))
                    y_min = int(m.group(2))
                    x_max = int(m.group(3))
                    y_max = int(m.group(4))
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)
        
        # 이미지 경로와 한 개 이상의 bounding box이 추출된 경우 data_list에 추가
        if img_path is not None and boxes:
            data_list.append({
                'img_path': img_path,
                'boxes': boxes,
                'labels': labels
            })
    
    return data_list


def get_region_proposals(img, scale=100, min_size=200):
    """
    selective search로 region proposals 반환
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, regions = selectivesearch.selective_search(img_rgb, scale=scale, min_size=min_size)

    proposals = []
    for r in regions:
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        proposals.append({'x': x, 'y': y, 'w': w, 'h': h})
    return proposals

def iou(boxA, boxB):
    """
    boxA, boxB: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def build_rcnn_entries(data_list):
    """
    data_list: load_penn_fudan_data()로 얻은 이미지별 GT 정보
    반환:
      images: OpenCV BGR 이미지를 리스트로 로딩 (각 인덱스로 접근)
      entries: [
          {
            'img_id': i,
            'region': (x, y, w, h),
            'label': 0 또는 1,
            'target_box': [x1, y1, x2, y2]
          },
          ...
      ]
    """
    images = []
    all_entries = []
    for i, item in enumerate(tqdm(data_list, desc="Building R-CNN entries")):
        img_path = item['img_path']
        gt_boxes = item['boxes']
        img = cv2.imread(img_path)
        images.append(img)
        if img is None:
            continue

        proposals = get_region_proposals(img, scale=100, min_size=200)
        for prop in tqdm(proposals, desc="Proposals", leave=False):
            x1p = prop['x']
            y1p = prop['y']
            x2p = x1p + prop['w']
            y2p = y1p + prop['h']
            prop_box = [x1p, y1p, x2p, y2p]

            best_iou = 0.0
            best_box = None
            for gbox in gt_boxes:
                val = iou(prop_box, gbox)
                if val > best_iou:
                    best_iou = val
                    best_box = gbox
            if best_iou >= 0.5:
                lbl = 1
                tbox = best_box
            else:
                lbl = 0
                tbox = [0, 0, 0, 0]

            entry = {
                'img_id': i,
                'region': (prop['x'], prop['y'], prop['w'], prop['h']),
                'label': lbl,
                'target_box': tbox
            }
            all_entries.append(entry)
    return images, all_entries

class RCNNProposalsDataset(Dataset):
    """
    PyTorch Dataset: region crop -> tensor
    """
    def __init__(self, images, entries, transform=None):
        self.images = images
        self.entries = entries
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img_id = e['img_id']
        x, y, w, h = e['region']
        label = e['label']
        tbox = e['target_box']

        img_bgr = self.images[img_id]
        crop = img_bgr[y:y+h, x:x+w, :]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor_img = self.transform(pil_img)
        return tensor_img, label, np.array(tbox, dtype=np.float32)
