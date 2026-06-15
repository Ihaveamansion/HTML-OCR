import random
from flask import json
from playwright.sync_api import sync_playwright
import os
from PIL import Image
import io
import numpy as np




def gen_string_p2(len):
    # Recursively generate a string of random characters
    # from the pool. The pool includes punctuation, lowercase
    # letters, and uppercase letters.
    pool=['!', ',', '?', '.',
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
          'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
          'u', 'v', 'w', 'x', 'y', 'z',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z',]
    c=random.choice(pool)
    if len==1:
        return c
    return gen_string_p2(len-1)+c

def gen_string_p1(len, pad_to):
    s=gen_string_p2(len)
    pad=pad_to-len
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

def render(html):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 1000})
        page.set_content(html)
        x1 = random.randint(0, 240)
        y1 = random.randint(0, 400)
        x2 = random.randint(760, 1000)
        y2 = random.randint(600, 1000)
        img=page.screenshot(
            clip={"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
        )
        browser.close()
        return img

def generate_rgb():
    rgb1=(random.randint(0,8),random.randint(0,8),random.randint(0,8))
    rgb2=(random.randint(0,8),random.randint(0,8),random.randint(0,8))
    while True:
        v = contrast_ratio(rgb1, rgb2)
        if v[0] >3:
            break
        rgb1=(random.randint(0,8),random.randint(0,8),random.randint(0,8))
        rgb2=(random.randint(0,8),random.randint(0,8),random.randint(0,8))
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
    font_dict=data["font_dict"]

    # Load existing npz data to check if it's corrupted,
    # Show if it is, and whether or not it is, ask if
    # the user wants to add on top of existing data
    # or generate completly new data
    try:
        data=np.load(np_path)
        for key in data.files:
            print(key, data[key].shape)
        texts=data["text"].tolist()
        ratios=data["ratio"].tolist()
        str_lns=data["str_ln"].tolist()
        font_ids=data["font_ids"].tolist()
        is_text_darkers=data["is_text_darker"].tolist()
        imgs=["imgs"].tolist()
    except:
        print("Npz not loaded.")
        texts=[]
        ratios=[]
        str_lns=[]
        font_ids=[]
        is_text_darkers=[]
        imgs=[]

    delete=input("Delete old data? y or 1 for yes")
    if delete=='1' or delete=='y':
        code=random.randint(0,10000)
        delete=int(input("Delete old data? Type in code " + str(code)+": "))
        if delete==code:
            print("Deleting")
            for path in os.listdir(img_path):
                os.remove(os.path.join(img_path,path))
            counter=0
            texts=[]
            ratios=[]
            str_lns=[]
            font_ids=[]
            is_text_darkers=[]
            imgs=[]
    

    for _ in range(num_pairs):
        if (_ + 1) % (num_pairs // 100) == 0:
            percent = (_ + 1) / num_pairs * 100
            print(f"{percent:.0f}%")
        # Randomly generate the text length, then the text,
        # then the text and background colors, then the font,
        # then render
        # All the text properties are kept track of in the
        # npz file, for error diagnosis,
        # and the images are saved in a separate folder.
        str_ln=random.randint(min_ln,max_ln)
        text=gen_string_p1(str_ln,max_ln)
        rgb1, rgb2, ratio, is_text_darker=generate_rgb()
        font=random.choice(fonts)
        font_id=font_dict[font]

        i=render(make_html(text,rgb1,rgb2,font))
        rgb=Image.open(io.BytesIO(i)).convert("RGB")
        rgb=rgb.resize((100,100))
        rgb=np.array(rgb)
        

        # save images to separate folder for error diagnosis,
        # but only if we have not already saved enough
        if _ < num_images:
            with open(f"{img_path}/image_{counter}.png", "wb") as f:
                f.write(i)

        # add everything to arrays to save to npz later
        texts.append(text)
        ratios.append(ratio)
        str_lns.append(str_ln)
        font_ids.append(font_id)
        is_text_darkers.append(is_text_darker)
        imgs.append(rgb)


        counter+=1

    np.savez_compressed(
        np_path,
        texts=np.array(texts),
        ratios=np.array(ratios),
        str_lns=np.array(str_lns),
        font_ids=np.array(font_ids),
        is_text_darkers=np.array(is_text_darkers),
        imgs=np.array(imgs)
    )
    with open(id_json, "w") as f:
        json.dump({"counter": counter, "font_dict": font_dict}, f)

if __name__ == "__main__":

    generate_image('imgs', 'imgs.npz', 
        'id.json', int(input("Min length of texts: ")),
        int(input("Max length of texts: ")),
        int(input("Number of pairs to generate: ")),
        int(input("Number of images to generate: ")))