from torch.utils.data import Dataset, DataLoader

import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pdb

class MyDataset(Dataset):

    def __init__(self, data_path:str, require_trace=True):
        self.data_path = data_path
        self.require_trace = require_trace

    def __len__(self):
        files = os.listdir(self.data_path)
        return sum(1 for file in files if os.path.isfile(os.path.join(self.data_path, file)))

    def __getitem__(self, i):
        # Open JSON 
        with open(os.path.join(self.data_path, 'data_'+str(i)+'.json'), "r") as f:
            data = json.load(f)
        # Load image features
        image_id = data['image_id']
        semantic_feature = torch.load(os.path.join(self.data_path,'semantic_feature', image_id+'.pt'))
        position_feature = torch.load(os.path.join(self.data_path,'position_feature', image_id+'.pt'))
        # Load text
        text = data['caption']
        # Load trace
        if self.require_trace:
            trace = list(data['bounding_box'].values())
            trace = torch.tensor(trace).float()
            return {'image_id': image_id, 'semantic_feature':semantic_feature, 'position_feature':position_feature,'text':text, 'trace':trace}
        else:
            return {'image_id': image_id, 'semantic_feature':semantic_feature, 'position_feature':position_feature, 'text':text}
    
    
def collate_func(batch, tokenizer, require_trace=False):
    image_ids = [item['image_id'] for item in batch]
    semantic_features = [item['semantic_feature'] for item in batch]
    position_features = [item['position_feature'] for item in batch]
    texts = [item['text'] for item in batch]
    encoded_batch = tokenizer.batch_encode_plus(
        texts,
        padding="longest",
        truncation=True,
        max_length=512, 
        return_tensors="pt"
    )
    text = encoded_batch["input_ids"]

    if require_trace:
        traces = [item['trace'] for item in batch]
        trace = nn.utils.rnn.pad_sequence(traces, batch_first=True)
        return image_ids, torch.stack(semantic_features), torch.stack(position_features), text, trace
    else:
        return image_ids, torch.stack(semantic_features), torch.stack(position_features), text



if __name__ == '__main__':
    
    """
    make sure that the flags of require_trace and require_cnn are matched in data_set and collate_fn
    if you are going to use images and texts: require_trace=False, require_cnn=False; Done√
    if you are going to use images, texts and boxes(coordinates and areas): require_trace=True, require_cnn=False; Done√
    if you are going to use images, texts, boxes (coordinates and areas) and trace images: require_trace=True, require_cnn=True; NB: this function is still under construction, DO NOT use it!
    """

    # an example to use dataloader

    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        ])
    
    Training_set = MyDataset(id_file="./TrainingSet/Data/img_list.txt",
                 pth_path="./TrainingSet/Data/",
                 image_path="./flickr30k_images/flickr30k_images/",
                 transform=transform,
                 require_trace=False,
                 require_cnn=False,
                 trace_path="./trace_images/train/")
    
    dataloader = DataLoader(Training_set,batch_size=16,shuffle=True,collate_fn=lambda batch: collate_func(batch, require_trace=False, require_cnn=False, transform=None))

    for image, text in dataloader:
        print(image.shape)
        print(text.shape)
        break


