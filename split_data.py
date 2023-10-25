import os
import json
import tqdm

with open('../flickr30k_train_localized_narratives_bb.jsonl') as json_file:
  for i, entry in enumerate(tqdm.tqdm(json_file)):
    old_data = json.loads(entry)
    data = {}
    data['image_id'] = old_data['image_id']
    data['caption'] = old_data['caption']
    data['bounding_box'] = old_data['bounding_box']
    with open(os.path.join('../train','data_'+str(i)+'.json'), 'w') as new_json:
      json.dump(data, new_json)