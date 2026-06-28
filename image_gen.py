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

NPZ_PATH='./npy/'


pool=[]
for i in range(65,91):
    pool.append(chr(i))
    
for i in range(97,123):
    pool.append(chr(i))

#uses input image id in order to generate image properties, with each id always the same random properties

def gen_img_prop(id, min_ln, max_ln, fonts):
    str_ln=np.random.default_rng(id*100).integers(min_ln,max_ln)
    
    text=gen_string_p1(str_ln, max_ln, np.random.default_rng(id*100+1))

    rgb1, rgb2, ratio, is_text_darker=generate_rgb(np.random.default_rng(id*100+2))

    font=fonts[np.random.default_rng(id*100+3).integers(0,len(fonts))]

    ss_rand=np.random.default_rng(id*100+4)

    x1 = int(ss_rand.integers(0, 240))
    y1 = int(ss_rand.integers(0, 400))
    x2 = int(ss_rand.integers(760, 1000))
    y2 = int(ss_rand.integers(600, 1000))

    return str_ln, text, rgb1, rgb2, ratio, is_text_darker, font, [x1,x2,y1,y2]


def gen_string_p2(ln, str_rand):
    # Recursively generate a string of rand characters
    # from the pool. The pool includes punctuation, lowercase
    # letters, and uppercase letters.
    c=pool[str_rand.integers(0, len(pool))]
    if ln==1:
        return c
    return (gen_string_p2(ln-1, str_rand)+c)

def gen_string_p1(ln, pad_to, str_rand):
    s=gen_string_p2(ln, str_rand)
    pad=pad_to-ln
    return s+chr(64)*pad

def rel_luminance(rgb):
    def f(c):
        c = c/8
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
    r, g, b = map(f, rgb)
    return 0.2126*r + 0.7152*g + 0.0722*b

def contrast_ratio(rgb1, rgb2):
    # Calculate the contrast ratio between two RGB colors using
    # the formula:
    # (L1 + 0.05) / (L2 + 0.05), where L1 is the relative
    # luminance of the lighter color and L2 is the relative
    # luminance of the darker color.
    L1 = rel_luminance(rgb1)
    L2 = rel_luminance(rgb2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    if L1>=L2:
        w=0   
    else: w=1
    # 1 means rgb1, or the text color, is darker than the
    # background color, and False means the opposite.
    # This is important for labeling the data correctly.
    return (lighter + 0.05) / (darker + 0.05), w

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

def generate_rgb(color_rand):
    rgb1=(color_rand.integers(0,8),color_rand.integers(0,8),color_rand.integers(0,8))
    rgb2=(color_rand.integers(0,8),color_rand.integers(0,8),color_rand.integers(0,8))
    while True:
        v = contrast_ratio(rgb1, rgb2)
        if v[0] >3:
            break
        rgb1=(color_rand.integers(0,8),color_rand.integers(0,8),color_rand.integers(0,8))
        rgb2=(color_rand.integers(0,8),color_rand.integers(0,8),color_rand.integers(0,8))
    return rgb1, rgb2, v[0], v[1]

def generate_image(img_path, min_ln, max_ln, id_start, id_end):
    imgs=[]
    labels=[]
    errors=[]
    ids=[]
    
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)
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
            errors.append(id)
            continue
        rgb = np.asarray(Image.open(io.BytesIO(i)).resize((200,200)))
        rgb = rgb.astype(np.uint8)
        rgb /= 32.0
        

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
    min_ln=int(input('Min text len: '))
    max_ln=int(input('Max text len: '))
    start=int(input('Num of pairs: '))
    end=int(input('Num of pairs: '))
    num_pairs=start-end

    os.makedirs(img_path, exist_ok=True)

    # Set manually or use multiprocessing.cpu_count()
    num_cores = multiprocessing.cpu_count()

    # Choose how many worker processes to use

    print(f"Detected cores: {num_cores}")

    workers=int(input("Num of workers: "))
    max_workers = min(workers, num_cores)

    print(f"Using workers: {max_workers}")
    print(f"Running {num_pairs} tasks\n")

    split=[]
    chunk=(num_pairs/max_workers)

    for i in range(max_workers):
        id_start=chunk*i+start
        split.append(
            [img_path,
            min_ln,
            max_ln,
            int(id_start),
            int(id_start+chunk),]
        )

    all_images=[]
    all_labels=[]
    all_errors=[]
    all_ids=[]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_image, *arg)
            for arg in split
        ]

        for future in as_completed(futures):
            result=future.result()
            if type(result) is str:
                print('fail')
                exit()
            imgs, label, errors, ids=result[0]
            imgs = np.concatenate(imgs)
            imgs = np.transpose(imgs, (0,3,1,2))
            labels = np.concatenate(labels)
            ids = np.concatenate(ids)
            all_ids.append(np.array(ids))
            np.savez(NPZ_PATH+f"{result[1][0]}-{result[1][1]}.npy", imgs=imgs, labels=labels, ids=ids)

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