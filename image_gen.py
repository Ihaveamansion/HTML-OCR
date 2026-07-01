import json
from playwright.sync_api import sync_playwright
import os
from PIL import Image
import io
import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import traceback
import psutil
from image_prop_utils import gen_img_prop

mem = psutil.virtual_memory()

#maximum number of images allowed to be on RAM before being saved in order to avoid hitting RAM cap, set as half as much RAM as when program starts
max_images=int(mem.available/(3*256*256)/2)
RESOLUTION=256
#Shard size each worker generates before saving to npz, set to 20k images per shard, which is about 5GB of RAM
#System is designed to run on a 9950x with 96GB of RAM, so 5GB is a safe limit to avoid hitting RAM cap
SHARD_SIZE=25000
#Limit number of times each worker attempts to generate the same shard
ATTEMPT_LIMIT=2

NPZ_PATH='./npz/'
os.makedirs('./npz',exist_ok=True)

def worker_manager(img_path, min_ln, max_ln, shard_start, shard_end):
    for shard in range(shard_start,shard_end):
        for attempt in range(ATTEMPT_LIMIT):
            print(f"Generating shard {shard} (attempt {attempt+1}/{ATTEMPT_LIMIT})")
            start_id=(shard)*SHARD_SIZE
            end_id=(shard+1)*SHARD_SIZE
            try:
                result=generate_image(img_path, min_ln, max_ln, start_id, end_id)
            except:
                traceback.print_exc()
                print(f'Shard {shard} failed')
                continue
            if type(result) is str:
                print('fail')
                exit()
            imgs, label, errors, ids=result[0]
            imgs=np.array(imgs)
            label=np.array(label)
            imgs = np.transpose(imgs, (0,3,1,2))
            all_ids.append(np.array(ids))
            path=NPZ_PATH+f"{result[1][0]}-{result[1][1]}.npz"
            print(path+' completed')
            print("Imgs shape: ", imgs.shape)
            print("Labels shape: ", label.shape)
            np.savez(path, imgs=imgs, labels=label, ids=ids)

def make_html(text,rgb1,rgb2,font):
    # Produce an HTML string that renders the given text with
    # the given text and background color, and font. The text
    # is centered in a 1000x1000 pixel div.
    l=''
    for letter in text:
        if ord(letter)==64:
            break
        l=l+letter
    return f"""
    <html>
    <body style="margin:0;">
        <div style="
            font-family:{font};
            font-size:50px;
            color:rgb({rgb1[0]*32},{rgb1[1]*32},{rgb1[2]*32});
            background:rgb({rgb2[0]*32},{rgb2[1]*32},{rgb2[2]*32});
            width:1000px;height:1000px;
            display:flex;align-items:center;justify-content:center;">
            {l}
        </div>
    </body>
    </html>
    """

def render(html, coords, page):
    page.set_content(html)
    img=page.screenshot(
        clip={"x": coords[0], "y": coords[2], "width": coords[1] - coords[0], "height": coords[3] - coords[2]}
    )
    return img


def generate_image(img_path, min_ln, max_ln, id_start, id_end):
    imgs=[]
    labels=[]
    errors=[]
    ids=[]
    
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True,
    args=[
    "--no-sandbox",
    "--disable-dev-shm-usage"
])
    page = browser.new_page(viewport={"width": 1000, "height": 1000})

    fonts=['Arial', 'Verdana', 'Helvetica', 'Times New Roman', 'Courier New', 'Tahoma', 'Trebuchet MS', 'Georgia', 'Garamond', 'Palatino Linotype','Courier New']

    for id in (range(id_start,id_end)):
        # randly generate the text length, then the text,
        # then the text and background colors, then the font,
        # then render
        # All the text properties are kept track of in the
        # npz file, for error diagnosis,
        # and the images are saved in a separate folder.
        
        prop=gen_img_prop(id,min_ln,max_ln,fonts)
        try:
            i=render(make_html(prop[1],prop[2],prop[3],prop[6]), prop[7], page)
        except:
            traceback.printexc()
            errors.append(id)
            continue
        rgb = np.asarray(Image.open(io.BytesIO(i)).resize((RESOLUTION,RESOLUTION)))
        rgb = rgb/32
        rgb = rgb.astype(np.uint8)
        

        # add everything to arrays to save to npz later
        imgs.append(rgb)
        label = [ord(item) for item in prop[1]]
        labels.append(np.array(label))
        ids.append(id)

    page.close()
    browser.close()
    p.stop()
    if (np.array(imgs).ndim)==(1):
        return 'fail'
    return [imgs, labels, errors, ids],[id_start,id_end]

if __name__=='__main__':
    img_path='imgs'
    #min_ln=int(input('Min text len: '))
    #max_ln=int(input('Max text len: '))
    #start=int(input('Start id: '))
    #end=int(input('End id: '))
    min_ln=1
    max_ln=20
    shard_start=0
    shard_end=200
    num_shards=shard_end-shard_start

    # Set manually or use multiprocessing.cpu_count()
    num_cores = multiprocessing.cpu_count()

    # Choose how many worker processes to use

    print(f"Detected cores: {num_cores}")

    #workers=int(input("Num of workers: "))
    workers=10
    max_workers = min(workers, num_cores)
    if max_workers*SHARD_SIZE*3*RESOLUTION*RESOLUTION > mem.available:
        print("Warning: The number of workers may exceed available RAM. Consider reducing other background processes, the number of workers, the shard limit to avoid running out of memory.")

    print(f"Using workers: {max_workers}")

    print(f"Running {num_shards*SHARD_SIZE} tasks\n")

    split=[]
    chunk=(num_shards/max_workers)

    for i in range(max_workers):
        s=chunk*i+shard_start
        split.append(
            [img_path,
            min_ln,
            max_ln,
            int(s),
            int(s+chunk),]
        )

    all_images=[]
    all_labels=[]
    all_errors=[]
    all_ids=[]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(worker_manager, *arg)
            for arg in split
        ]

"""    imgs = np.concatenate(all_images)
    labels = np.concatenate(all_labels)
    ids = np.concatenate(all_ids)

    order = np.argsort(ids)
    imgs = imgs[order]
    labels = labels[order]
    ids = ids[order]

    print(labels)
    imgs = np.transpose(imgs, (0,3,1,2))
    print(imgs.shape)
    print(labels.shape)
    print(ids.shape)
    print(all_errors)
    with open('errors.json', 'w') as f:
        json.dump(all_errors, f)"""