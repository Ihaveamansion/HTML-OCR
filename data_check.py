import numpy as np
from PIL import Image
import json
import os

NP_PATH='npz/0-8.npz'
CHECK_PATH='./check'

os.makedirs(CHECK_PATH, exist_ok=True)
data=np.load(NP_PATH)

imgs=data['imgs']
labels=data['labels']
ids=data['ids']


print(imgs.shape)
print(labels.shape)
print(ids.shape)

check=int(input('Num of pairs to check: '))
words=[]
id=[]

for i in range(check):
    image=np.transpose(imgs[i], (1,2,0))
    img = (image * 32).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(CHECK_PATH+'/'+str(ids[i])+".png")
    label=labels[i]
    words.append('')
    id.append(ids[i].astype(np.ndarray))
    for l in label:
        if l==64:
            break
        words[i]+=chr(l)

with open((CHECK_PATH+'/words.json'), 'w') as f:
    json.dump(words, f)

with open((CHECK_PATH+'/id.json'), 'w') as f:
    json.dump(id, f)
