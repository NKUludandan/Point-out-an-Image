import os
import json
import tqdm

import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image

import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define ResNet
resnet = models.resnet50(pretrained=True).to(device)
resnet = nn.Sequential(*list(resnet.children())[:-1])
for param in resnet.parameters():
    param.requires_grad = False
# Define FRCNN
frcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
for param in frcnn.parameters():
    param.requires_grad = False

resnet_transform = transforms.Compose([
    transforms.Resize((232,232)),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
frcnn_transform = transforms.Compose([
    transforms.ToTensor(),
])
save_path = '../val'


with open('../flickr30k_val_localized_narratives_bb.jsonl') as json_file:
  for i, entry in enumerate(tqdm.tqdm(json_file)):
    data = json.loads(entry)
    image_id = data['image_id']
    # Check if the file exists in the folder
    if (os.path.isfile(os.path.join(save_path,'semantic_feature',image_id+'.pt'))) and (os.path.isfile(os.path.join(save_path,'position_feature',image_id+'.pt'))):
        continue
    image = Image.open(os.path.join('../flickr30k_images/flickr30k_images',image_id+'.jpg')).convert("RGB")
    frcnn_image = frcnn_transform(image).unsqueeze(0).to(device)
    # Get features
    frcnn.eval()
    with torch.no_grad():
      local_features  = frcnn(frcnn_image)
    resnet_image = resnet_transform(frcnn_image)
    resnet.eval()
    with torch.no_grad():
      global_feature = resnet(resnet_image).view(-1, 2048)
    # Select the top 16 bounding boxes
    frcnn_image = frcnn_image.squeeze(0)
    height, width = frcnn_image.size(1), frcnn_image.size(2)
    num_boxes = min(16, local_features[0]['boxes'].shape[0])
    selected_boxes = local_features[0]['boxes'][:num_boxes]
    # Crop the image patch
    crop_boxes = selected_boxes.int()
    # If no objects found
    try:
        image_patch = torch.stack([resnet_transform(frcnn_image[:, crop_boxes[j,1]:crop_boxes[j,3], crop_boxes[j,0]:crop_boxes[j,2]]) for j in range(crop_boxes.size(0))])
        # Compute local semantic feature
        resnet.eval()
        with torch.no_grad():
            local_feature = resnet(image_patch).view(-1, 2048)
    except:
        local_feature = torch.zeros(16, 2048).to(selected_boxes.device)
        selected_boxes = torch.zeros(16, 4).to(selected_boxes.device)

    # Pad with 0
    if (num_boxes < 16) and (num_boxes != 0):
      padding_boxes = torch.zeros(16 - num_boxes, 4).to(selected_boxes.device)
      padding_regions = torch.zeros(16 - num_boxes, 2048).to(selected_boxes.device)
      selected_boxes = torch.cat((selected_boxes, padding_boxes))
      local_feature = torch.cat((local_feature, padding_regions))
    # Normalization
    selected_boxes[:, [0, 2]] /= width
    selected_boxes[:, [1, 3]] /= height
    # Add area
    area = (selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1])
    selected_boxes = torch.cat([selected_boxes, area.unsqueeze(1)], dim=1)
    selected_boxes[:, [1, 2]] = selected_boxes[:, [2, 1]] # need to be xmin, xmax, ymin, ymax
    # Combine local and global features
    semantic_feature = torch.cat((global_feature, local_feature))
    # Combine local and global positions
    position_feature = torch.cat((torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]).unsqueeze(0).to(selected_boxes.device), selected_boxes))
    # Save as pth
    torch.save(semantic_feature, os.path.join(save_path,'semantic_feature',image_id+'.pt'))
    torch.save(position_feature, os.path.join(save_path,'position_feature',image_id+'.pt'))
# pt_files = os.listdir('../val/position_feature')
# for file in pt_files:
#     position_feature = torch.load(os.path.join('../val/position_feature', file))
#     if position_feature.size(0) != 17:
#         print("File is wrong")
# print("All Ok")
