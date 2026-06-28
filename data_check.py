import numpy as np
from PIL import Image

NP_PATH='100.npz'
CHECK_PATH='./check'

data=np.load(NP_PATH)

imgs=data['imgs']
labels=data['labels']
ids=data['ids']


print(imgs.shape)
print(labels.shape)
print(ids.shape)

check=int(input('Num of pairs to check: '))
words=[]

for i in range(check):
    image=np.transpose(imgs[i], (1,2,0))
    image*=32
    img = Image.fromarray(image)
    img.save(CHECK_PATH+'/'+str(ids[i])+".png")
    label=labels[i]
    words.append('')
    for l in label:
        if l==64:
            break
        words[i]+=chr(l)
with open((CHECK_PATH+'/words'), 'w') as f:
    json.dump(words, f)