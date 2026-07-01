import numpy as np


#uses input image id in order to generate image properties, with each id always the same random properties

pool=[]
for i in range(65,91):
    pool.append(chr(i))
    
for i in range(97,123):
    pool.append(chr(i))

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