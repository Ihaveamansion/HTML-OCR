from flask import json
from playwright.sync_api import sync_playwright
import os
from PIL import Image
import io
import numpy as np
import tqdm

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
    pool=['!', ',', '?', '.',
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
          'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
          'u', 'v', 'w', 'x', 'y', 'z',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z',]
    c=pool[str_rand.integers(0, len(pool))]
    if ln==1:
        return c
    return (gen_string_p2(ln-1, str_rand)+c)

def gen_string_p1(ln, pad_to, str_rand):
    s=gen_string_p2(ln, str_rand)
    pad=pad_to-ln
    return s+'_'*pad

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
        if letter=='_':
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

def render(html, coords):
    page = browser.new_page(viewport={"width": 1000, "height": 1000})
    page.set_content(html)
    img=page.screenshot(
        clip={"x": coords[0], "y": coords[2], "width": coords[1] - coords[0], "height": coords[3] - coords[2]}
    )
    page.close()
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

def generate_image(img_path, np_path, id_json, min_ln, max_ln,
                    num_pairs, num_images):

    if num_pairs < num_images:
        print("Number of pairs must be greater than or equal to number of images.")
        return

    os.makedirs(img_path, exist_ok=True)

    fonts=['Arial', 'Verdana', 'Helvetica', 'Times New Roman', 'Courier New',
           'Tahoma', 'Trebuchet MS', 'Georgia', 'Garamond', 'Palatino Linotype',
           'Courier New']

    # The purpose of the json file is so that we can keep track of
    # the font ids and counter
    with open(id_json, "r") as f:
        data = json.load(f)
    counter=data["counter"]

    # Load existing npz data to check if it's corrupted,
    # Show if it is, and whether or not it is, ask if
    # the user wants to add on top of existing data
    # or generate completly new data
    try:
        data=np.load(np_path)
        for key in data.files:
            print(key, data[key].shape)
        imgs=data["imgs"].tolist()
        labels=data["labels"].tolist()
    except:
        print("Npz not loaded.")
        imgs=[]
        labels=[]

    delete=input("Delete old data? y or 1 for yes")
    # To prevent accidental deletion of data, we ask the user to type in a
    # random code before deleting the old data. If the user does not type
    # in the correct code, the old data will not be deleted and the new
    # data will be added on top of it.
    if delete=='1' or delete=='y':
        code=rng.integers(1000,9999)
        delete=int(input("Delete old data? Type in code " + str(code)+": "))
        if delete==code:
            print("Deleting")
            for path in os.listdir(img_path):
                os.remove(os.path.join(img_path,path))
            counter=0
            imgs=[]
            labels=[]
    

    for _ in tqdm.tqdm(range(num_pairs)):
        # randly generate the text length, then the text,
        # then the text and background colors, then the font,
        # then render
        # All the text properties are kept track of in the
        # npz file, for error diagnosis,
        # and the images are saved in a separate folder.
        
        prop=gen_img_prop(counter,min_ln,max_ln,fonts)

        i=render(make_html(prop[1],prop[2],prop[3],prop[6]), prop[7])
        rgb=Image.open(io.BytesIO(i)).convert("RGB")
        rgb=rgb.resize((100,100))
        rgb=np.array(rgb)
        

        # save images to separate folder for error diagnosis,
        # but only if we have not already saved enough
        if _ < num_images:
            with open(f"{img_path}/image_{counter}.png", "wb") as f:
                f.write(i)

        # add everything to arrays to save to npz later
        imgs.append(rgb.ravel())
        label = [ord(item) for item in prop[1]]
        labels.append(label)


        counter+=1

    np.savez_compressed(
        np_path,
        imgs=np.array(imgs),
        labels=np.array(labels)
    )
    with open(id_json, "w") as f:
        json.dump({"counter": counter}, f)

if __name__ == "__main__":
    rng=np.random.default_rng()

    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)

    generate_image('imgs', 'imgs.npz', 
        'id.json', int(input("Min length of texts: ")),
        int(input("Max length of texts: ")),
        int(input("Number of pairs to generate: ")),
        int(input("Number of images to generate: ")))
    
    browser.close()
    p.stop()