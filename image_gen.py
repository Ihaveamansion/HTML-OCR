import argparse
import math
import time

from playwright.sync_api import sync_playwright
import os
from PIL import Image
import io
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Array, Lock, Value, Manager
import traceback
import psutil
from image_prop_utils import gen_img_prop
import csv

mem = psutil.virtual_memory()

#maximum number of images allowed to be on RAM before being saved in order to avoid hitting RAM cap, set as half as much RAM as when program starts
max_images=int(mem.available/(3*256*256)/2)
RESOLUTION=256
#Shard size each worker generates before saving to npz, set to 20k images per shard, which is about 5GB of RAM
#System is designed to run on a 9950x with 96GB of RAM, so 5GB is a safe limit to avoid hitting RAM cap
SHARD_SIZE=100
#Limit number of times each worker attempts to generate the same shard
ATTEMPT_LIMIT=2

MIN_LN=1
MAX_LN=20
SHARD_START=0
SHARD_END=100
max_workers=10
PAD_TOKEN=64

CLOCK = None
LOCK = None
PROGRESS = None

def increment_clock(v):
    CLOCK.value += 1
    if v==1:
        CLOCK.value = -(max_workers)-1


def init(clock, lock, progress):
    global CLOCK, LOCK, PROGRESS
    PROGRESS = progress
    CLOCK = clock
    LOCK = lock

NPZ_PATH='./npz/'
os.makedirs(NPZ_PATH,exist_ok=True)

def worker_manager(min_ln, max_ln, shard_start, shard_end, worker_id, 
                   tick, out):
    os.makedirs(out, exist_ok=True)
    for shard in range(shard_start,shard_end):
        for attempt in range(ATTEMPT_LIMIT):
            print(f"Generating shard {shard} (attempt {attempt+1}/{ATTEMPT_LIMIT})")
            current_struct = time.localtime()
            formatted_date = time.strftime("%Y-%m-%d %H:%M:%S", current_struct)
            print(f"Time is: {formatted_date}")
            start_id=(shard)*SHARD_SIZE
            end_id=(shard+1)*SHARD_SIZE
            try:
                result=generate_image(min_ln, max_ln, start_id, end_id, worker_id, tick, (shard-shard_start)*SHARD_SIZE, out)
            except:
                traceback.print_exc()
                print(f'Shard {shard} failed')
                continue
            if type(result) is str:
                print('fail')
                raise RuntimeError("Shard failed")
                continue
            imgs, label, ids=result[0]
            imgs=np.asarray(imgs)
            label=np.asarray(label)
            imgs = np.transpose(imgs, (0,3,1,2))
            path=NPZ_PATH+f"{result[1][0]}-{result[1][1]}.npz"
            print(f'Shard {shard} completed')
            current_struct = time.localtime()
            formatted_date = time.strftime("%Y-%m-%d %H:%M:%S", current_struct)
            print(f"Time is: {formatted_date}")
            np.savez(path, imgs=imgs, labels=label, ids=ids)
            del imgs
            del label
            break
    with LOCK:
        PROGRESS[worker_id] = 1
        increment_clock(1)

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


def generate_image(min_ln, max_ln, id_start, id_end, worker_id, tick, worker_progress, out):
    imgs = []
    labels=np.full(
    (SHARD_SIZE,MAX_LN),
    PAD_TOKEN,
    dtype=np.uint8
)
    errors=[]
    ids=[]
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True,
    args=[
    "--no-sandbox",
    "--disable-dev-shm-usage"
])
    page = browser.new_page(viewport={"width": 1000, "height": 1000})

    fonts=['Arial', 'Verdana', 'Helvetica', 'Times New Roman', 'Courier New', 'Tahoma', 'Trebuchet MS', 'Georgia', 'Garamond', 'Palatino Linotype']

    for id in (range(id_start,id_end)):
        if (id-id_start+worker_progress>=((CLOCK.value*tick)+(worker_id*tick)))and (CLOCK.value!=(-(max_workers)-1)):
            with LOCK:
                PROGRESS[worker_id] = 1
            while True:
                if PROGRESS[worker_id]==0 or CLOCK.value==(-(max_workers)-1):
                    break
                if worker_id==0 and sum(PROGRESS)>=max_workers:
                    increment_clock(0)
                    for i in range(max_workers):
                        PROGRESS[i]=0
                time.sleep(1)
        
        # randomly generate the text length, then the text,
        # then the text and background colors, then the font,
        # then render
        # All the text properties are reproducable from a random
        # known seed, so everything can be recalculated quickly without wasting a lot of storage
        
        prop=gen_img_prop(id,min_ln,max_ln,fonts)
        try:
            i=render(make_html(prop[1],prop[2],prop[3],prop[6]), prop[7], page)
        except:
            traceback.print_exc()
            errors.append(id)
            continue
        img = Image.open(io.BytesIO(i))
        del i
        img = img.resize((RESOLUTION, RESOLUTION))
        rgb = np.array(img, dtype=np.uint8)
        del img
        rgb //=32
        rgb = rgb.astype(np.uint8)
        

        # add everything to arrays to save to npz later
        imgs.append(rgb)
        label = [ord(item) for item in prop[1]]
        labels[id-id_start,:len(label)] = label
        ids.append(id)
        current_struct = time.localtime()
        formatted_date = time.strftime("%Y-%m-%d %H:%M:%S", current_struct)
        with open(out+'/'+out+'-'+str(worker_id)+'.csv', mode="a", newline="", encoding="utf-8") as file:
            writer=csv.writer(file)
            writer.writerow([formatted_date, len(imgs), len(labels)])

    page.close()
    browser.close()
    p.stop()
    if len(ids) == 0:
        return 'fail'
    return [[np.array(imgs, dtype=np.uint8), labels, ids], [id_start, id_end]]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Generate images of text with random properties.',
                                   prog='image_gen.py')
    parser.add_argument('--staggered', action='store_true')
    parser.add_argument('-start', default=SHARD_START, type=int)
    parser.add_argument('-end', default=SHARD_END, type=int)
    parser.add_argument("output_file", default="gen.csv")
    args=parser.parse_args()
    SHARD_START=args.start
    SHARD_END=args.end
    if args.staggered:
        clock_init=-(max_workers)
    else:
        clock_init=-max_workers-1
    multiprocessing.set_start_method("spawn", force=True)
    #min_ln=int(input('Min text len: '))
    #max_ln=int(input('Max text len: '))
    #start=int(input('Start id: '))
    #end=int(input('End id: '))
    min_ln=MIN_LN
    max_ln=MAX_LN
    shard_start=SHARD_START
    shard_end=SHARD_END
    num_shards=shard_end-shard_start

    num_cores = os.cpu_count()

    # Choose how many worker processes to use

    print(f"Detected cores: {num_cores}")

    if num_cores < max_workers:
        exit(f"Error: Detected {num_cores} cores, but max_workers is set to {max_workers}. Please reduce max_workers to {num_cores} or less.")

    print(f"Using workers: {max_workers}")

    print(f"Running {num_shards*SHARD_SIZE} tasks\n")

    split=[]
    chunk=(num_shards/max_workers)
    chunk = (num_shards / max_workers)

    #Keeps track of each worker's progress in order to keep workers staggered and better utilize RAM
    progress = Array('i', [0]*max_workers)
    #Length of each 'tick'
    tick = SHARD_SIZE // max_workers
    #Tells which cycle each worker should be on
    clock = multiprocessing.Value('i', clock_init)
    lock = multiprocessing.Lock()

    for i in range(max_workers):
        start = round(shard_start + i * chunk)
        end = round(min(start + chunk, shard_end))

        if start >= end:
            break
        split.append(
            [min_ln,
            max_ln,
            int(start),
            int(end),
            i,
            tick,
            args.output_file]
        )
    print(max_workers)
    with ProcessPoolExecutor(
    max_workers=max_workers,
    initializer=init,
    initargs=(clock,lock, progress)
) as executor:
        futures = [
            executor.submit(worker_manager, *arg)
            for arg in split
        ]
        for future in as_completed(futures):
            future.result()  # re-raises any worker exception in the main process