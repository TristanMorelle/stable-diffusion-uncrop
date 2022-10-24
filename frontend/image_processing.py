"""
Written by Tristan Morelle 2022
The goal was to provide methods for filling and (un)cropping images
I opted to use pillow PIL for all operations to keep performance fast and avoid numpy calculations and convertions.
"""

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random, os, pickle

def data_as_rows(img):
    """
    ====================================================================================================================
    get image as rows
    :param img: source
    :return: list with rows of tuples representing pixel data
    ====================================================================================================================
    """
    data = list(img.getdata())
    img_w, img_h = img.size
    return [data[x:x + img_w] for x in range(0, len(data), img_w)]


def add_random_noise(pixel_value, noise_var = 5):
    """
    ====================================================================================================================
    manual color noise manually
    :param pixel_value: a tuple to add variation to
    :param noise_var:
    :return: modiefied tuple
    ====================================================================================================================
    """
    var = [random.randint(-5, noise_var) for i in range (len(pixel_value))]
    return tuple(pixel_value[i] + var[i] if pixel_value[i] + var[i] <= 255 else 255 for i in range(len(pixel_value)))


def add_margin(img, margins =(0,0,0,0), color = None):
    """
    extend image with borders
    ====================================================================================================================
    :param img: source image
    :param top: px amount to add on top
    :param right: px amount to add on right
    :param bottom: px amount to add on bottom
    :param left: px amount to add on left
    :param color: fill color
    :return: pil image
    ====================================================================================================================
    """
    top, right, bottom, left = margins
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result


def resize_canvas(img, width, height, bg_color = None):
    """
    resize canvas by pasting the prev image at the center
    ====================================================================================================================
    :param img:
    :param width:
    :param height:
    :param bg_color:
    :return: pil image
    ====================================================================================================================
    """
    if bg_color == 'auto':
        bg_color = get_bg_color(img, use_edge = False)

    elif bg_color == 'auto_edge':
        bg_color = get_bg_color(img, use_edge=True)

    elif bg_color != None:
        bg_color = bg_color
    else:
        bg_color = (255, 255, 255, 0)

    img = img.convert('RGBA')
    background = Image.new('RGBA', (width, height), bg_color)
    bg_w, bg_h = background.size
    img_w, img_h = img.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset, img )
    return background


def sigmoid_range(pos=0.0, slope=0.0, x_range=1.0, normalize_neg=False):
    """
    versitile function for a sigmoid activation curve with variable slopes and range
    can be used linear as well as smoothed
    ====================================================================================================================
    :param pos: position along the X axis
    :param slope: 0 is linear, 1 is inverted S, -1 is S
    :param x_range: the max range the pos is traveling over
    :param normalize_neg: output between -1 and 1 else => 0 to 1
    :return: result value given the function between 0 and 1 or -1 and 1 with normalize_neg
    ====================================================================================================================
    """
    if slope >= 1 or slope <= -1:
        raise RuntimeError('slope must be between -0.99 and 0.99')

    norm_range = 1.0 / x_range

    k = float(slope)
    x = float(pos * norm_range)

    if x > 1 or x < -1:
        raise RuntimeError('pos outside of range', pos, x_range)

    norm = (x - (k * x)) / (k - (2.0 * k * abs(x)) + 1)

    if normalize_neg:
        norm = (((norm + 1.0) * 1.0) / 2.0)
    return norm


def get_edge(img):
    """
    get a nice edge trace from the image
    ====================================================================================================================
    :param img: source image
    :return: pil image
    ====================================================================================================================
    """
    contour_a = img.split()[-1].filter(ImageFilter.FIND_EDGES)
    contour = img.convert("RGB")
    contour.putalpha(contour_a)
    return contour


def dither (img, opacity = 0.1, color = (0, 0, 0, 0), dark_noise= False):
    """
    ====================================================================================================================
    :param img:
    :param opacity:
    :param color:
    :return: pil image
    ====================================================================================================================
    """
    noise_a = Image.effect_noise(size = (img.size), sigma= 100).split()[-1]
    if dark_noise:
        # clip blacks and sharpen
        noise_a = ImageEnhance.Brightness(noise_a).enhance(0.05) # 0.0005 is min
        noise_a = ImageEnhance.Sharpness(noise_a).enhance(3)
        noise_a = ImageEnhance.Contrast(noise_a).enhance(10*opacity) # boosted contrast
    else:
        noise_a = ImageEnhance.Brightness(noise_a).enhance(opacity)


    # get rid of non blacks
    add_noise = Image.new(mode = 'RGBA', size = img.size, color=color)
    add_noise.putalpha(noise_a)
    img = Image.composite(add_noise, img, add_noise)
    return img


def set_brightness (img, opacity = 1.0):
    """
    ====================================================================================================================
    :param img:
    :param opacity:
    :return: pil image
    ====================================================================================================================
    """
    return ImageEnhance.Brightness(img).enhance(opacity)


def make_blurred_mask(img, mask_blur_range, mask_bright, dither_opacity, dither_color = None):
    """
    use content mask and blur it
    ====================================================================================================================
    :param img: source
    :param mask_blur_range: pixel distance to blur
    :param mask_bright: boost/ clip mask
    :param dither_opacity: how much dither noise gets added
    :param dither_color: color tu use for dither if any given
    :return: Pil Image
    ====================================================================================================================
    """
    # use mask of start image to create a blend mask to blurred bg
    mask = img.split()[-1]

    if (mask_blur_range % 2) == 0:
        mask_blur_range = int( mask_blur_range +1)
    mask = mask.filter(ImageFilter.MaxFilter(mask_blur_range))
    mask = mask.filter(filter=ImageFilter.GaussianBlur(mask_blur_range))
    mask = ImageEnhance.Brightness(mask).enhance(mask_bright)
    # mask = ImageEnhance.Contrast(mask).enhance(2.5)

    # og_mask = ImageOps.invert(og_mask)
    if dither_color != None:
        mask_dither = dither(img=mask, opacity=dither_opacity, color=dither_color)
        # mask dither to prevent lightening effect
        mask.paste(mask_dither, mask = mask)
    return mask


def put_color_border (img, bg_color = 50, size = (100, 10, 10, 10)):
    """
    paste border at the edge of the image
    ====================================================================================================================
    :param img:
    :param bg_color:
    :return:pil image
    ====================================================================================================================
    """
    if img == None:
        print ('no image', img)
        return img

    if type(size) in [int, float]:
        l_size, t_size, r_size, b_size = int(size),int(size),int(size),int(size)
    elif type(size) in [tuple, list]:
        l_size, t_size, r_size, b_size  = size
    else:
        raise RuntimeError("unknown type for size", type (size), size)

    img_w, img_h = img.size
    t = Image.new(mode='RGB', size=(img_w, t_size), color=bg_color)
    l = Image.new(mode='RGB', size=(l_size, img_w), color=bg_color)
    b = Image.new(mode='RGB', size=(img_w, b_size), color=bg_color)
    r = Image.new(mode='RGB', size=(r_size, img_w), color=bg_color)

    img.paste(t, box = (0,0)) #top
    img.paste(b, box = (0,img_h - b_size)) #bot

    img.paste(l, box = (0,0)) #left
    img.paste(r, box = (img_w - r_size, 0)) #right
    return img


def repeat_edges (img,
                  fill_border_size = None,
                  width = None,
                  height =None,
                  sample=1,
                  bg_color = None,
                  pre_fill_square = None,
                  shrink_faded_edges=0,
                  mask_final=True,
                  mask_blur_range=100,
                  mask_bright=2,
                  mask_dither_opacity = 0.01,
                  debug=False
                  ):
    """
    grow the borders inward to the actual image content (empty borders of black / alpha get cropped)
    ====================================================================================================================
    :param img: image to convert
    :param fill_border_size: custom l_size, t_size, r_size, b_size to fill with repeated edge, best to put None
    :param width: result canvas width
    :param height: result canvas height
    :param sample: size of pixels to sample
    :param bg_color: color to fill bg with  => color tuple, None, auto, auto_edge
    :param pre_fill_square: make image content square before repeating edges, filling => color tuple, auto, auto_edge, None
    :param shrink_faded_edges: helps to get rid of faded borders in images alpha's.
    :param mask_final: create mask based on the content and fade repeated edges with this mask
    :param mask_blur_range: mask blur amount
    :param mask_bright: mark brightness (can be usefull to clip the mask)
    :param mask_dither_opacity: add noise to mask to avoid banding effect
    :param debug: show inbetween steps
    :return: pil image
    ====================================================================================================================
    """

    if img == None:
        print('no image', img)
        return img

    if width != None and height != None:
        # resize canvas and keep current image size, bg color cant be set now to allow edge detection for repeat
        img = resize_canvas(img =img, width = width, height=height, bg_color=None)
    else:
        # dont resize
        width, height = img.size

    og_image = img.copy()

    img = img.convert('RGBA')
    img_w, img_h = img.size

    # get sample rows with actual content
    s_l, s_t, s_r, s_b = img.getbbox()

    # if fill content square or shrink edges
    if shrink_faded_edges or pre_fill_square != None:
        # get content
        content = img.crop((s_l, s_t, s_r, s_b))
        if pre_fill_square != None: # for speed
            if pre_fill_square == 'auto':
                pre_fill_square = get_bg_color(og_image, use_edge = False)

            elif pre_fill_square == 'auto_edge':
                pre_fill_square = get_bg_color(og_image, use_edge=True)

            elif pre_fill_square != None:
                pre_fill_square = pre_fill_square
            else:
                pre_fill_square = (255, 255, 255, 0)

            bg = Image.new(mode = "RGBA", size = content.size, color = pre_fill_square)
        else:
            bg = Image.new(mode="RGBA", size=content.size)

        if shrink_faded_edges:
            # crop mask for premultiplied edges
            if (shrink_faded_edges % 2) == 0:
                shrink_faded_edges = shrink_faded_edges + 1

            mask = content.split()[-1]
            mask = mask.filter(ImageFilter.MinFilter(shrink_faded_edges))
            # fill with edge cropped mask
            bg.paste(content, (0, 0), mask)
        else:
            # fill with optional bg color
            bg.paste(content, (0,0), content)

        # paste result on new bg
        img.paste(bg, (s_l, s_t), bg)

        if debug:
            img.show()

    # if given border size, else automatic
    if type(fill_border_size).__name__ in ('tuple', 'list') \
            and len (fill_border_size) == 4: # custom value for all sides
        l_size, t_size, r_size, b_size  = fill_border_size
    elif type(fill_border_size).__name__ in ('int', 'float') \
            and fill_border_size >= 0: # single input over all sides
        l_size, t_size, r_size, b_size = fill_border_size,fill_border_size,fill_border_size,fill_border_size
    else: #automatic size
        l_size, t_size, r_size, b_size  = s_l, s_t, img_w - s_r, img_h - s_b

    if all(i == 0 for i in (l_size, t_size, r_size, b_size)): # if zero, skip procedure to add borders
        # if no borders needed return (ie. zoom in)
        print ('no borders',(l_size, t_size, r_size, b_size))
    else:
        # make borders
        # make sure size is not 0
        l_size, t_size, r_size, b_size = [i if i else 1 for i in (l_size, t_size, r_size, b_size)]

        # get samples to paste in repeated area
        t = img.crop((0,s_t,img_w, s_t + sample))
        l = img.crop((s_l,0, s_l + sample, img_h))
        b = img.crop((0,s_b-sample, img_w, s_b ))
        r = img.crop((s_r - sample ,0,s_r, img_h))

        # get corner sample colors of 1*1 size
        t_l = img.crop((s_l,s_t,s_l +sample,s_t +sample))
        t_r = img.crop((s_r -sample, s_t,s_r, s_t +sample))
        b_l = img.crop((s_l,s_b - sample, s_l+sample, s_b))
        b_r = img.crop((s_r - sample, s_b -sample ,s_r, s_b))

        # scale borders
        t = t.resize((img_w, t_size), resample=Image.Resampling.LANCZOS)
        l = l.resize((l_size, img_h), resample=Image.Resampling.LANCZOS)
        b = b.resize((img_w, b_size), resample=Image.Resampling.LANCZOS)
        r = r.resize((r_size, img_h), resample=Image.Resampling.LANCZOS)

        # scale corners
        t_l = t_l.resize((l_size, t_size), resample=Image.Resampling.LANCZOS)
        t_r = t_r.resize((r_size, t_size), resample=Image.Resampling.LANCZOS)
        b_l = b_l.resize((l_size, b_size), resample=Image.Resampling.LANCZOS)
        b_r = b_r.resize((r_size, b_size), resample=Image.Resampling.LANCZOS)

        # fill
        img.paste(t, box = (0,0)) #top
        img.paste(b, box = (0,img_h - b_size)) #bot

        img.paste(l, box = (0,0)) #left
        img.paste(r, box = (img_w - r_size, 0)) #right

        # # corners
        img.paste(t_l, box = (0,0)) #topleft
        img.paste(t_r, box = (img_w - r_size, 0)) #top right

        img.paste(b_l, box = (0, img_h - b_size)) #botleft
        img.paste(b_r, box = (img_w - r_size, img_h - b_size)) #botright

    # if fill bg color
    if bg_color != None:
        if bg_color == 'auto':
            bg_color = get_bg_color(og_image, use_edge = False)

        elif bg_color == 'auto_edge':
            bg_color = get_bg_color(og_image, use_edge=True)

        elif bg_color != None:
            bg_color = bg_color

        else:
            bg_color = (255, 255, 255, 0)

        background = Image.new('RGBA', img.size, bg_color)
        background.paste(img, (0,0), img)
        img = background

    # if mask result
    if mask_final and mask_dither_opacity:
        mask = make_blurred_mask(img=og_image,
                                 mask_blur_range=mask_blur_range,
                                 mask_bright=mask_bright,
                                 dither_opacity=mask_dither_opacity,
                                 dither_color=bg_color)
        if debug:
            mask.show()
        fill = Image.new('RGBA', (width, height), bg_color)
        img = Image.composite(img, fill, mask)

    return img


def get_bg_color(img, use_edge = True):
    """
    ====================================================================================================================
    :param img: source image with alpha
    :param use_edge: use the image edge to calculate most occuring color
    get the most occuring bg color out of the edges, or general image
    ====================================================================================================================
    """
    # contour = img.filter(ImageFilter.ModeFilter(size=15))
    img = img.convert('RGBA')
    if use_edge:
        img = get_edge(img = img)

    all_colors = img.getcolors(img.size[0] * img.size[1])
    all_colors = sorted(all_colors, key=lambda x: x[0], reverse=True)

    if use_edge: # if use edge
        for count, v in all_colors:
            a = v[-1]
            if a >= 128:
                print("found contour", v, count)
                v = list(v)
                v[-1] = 255
                return tuple(v)
        else:
            # if nothing found continue on global image
            use_edge = False

    if not use_edge: # if use full image or didn't find anyrhing in edge
        for count, v in all_colors:
            a = v[-1]
            if a >= 128:
                print("no contour", v, count)
                v = list(v)
                v[-1] = 255
                return tuple(v)
        else:
            # finally if nothing was found return the first of count
            v = list(all_colors[0])
            v[-1] = 255
            return tuple(v)


def fill_content_proportionate(img, width, height, resize_canvas = True):
    """
    scale proportionally to the max output size
    ====================================================================================================================
    :param img: source image with alpha
    :param width: output canvas width
    :param height: output canvas height
    ====================================================================================================================
    """
    img_w, img_h = img.size
    delta_w = width - img_w
    delta_h = height - img_h
    # print('delta_w',  width, img_w, delta_w)
    # print('delta_h',  height, img_h, delta_h)

    # smallest delta needs to get added to both
    if delta_h <= delta_w:
        scale = float(height) / float(img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
    else:
        scale = float(width) / float(img_w)
        new_w = int(round(img_w * scale, 0))
        new_h = int(round(img_h * scale, 0))
    # print('prev size', img_w, img_h )
    # print ('new size', new_w, new_h)

    prop = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    if resize_canvas:
        bg = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        offset = ((width - new_w) // 2, (height - new_h) // 2)
        bg.paste(prop, offset, prop)
        prop = bg
    return prop


def crop_content(img, invert = True):
    """
    scale proportionally to the max output size using the content bounding box
    ====================================================================================================================
    :param img: source image with alpha
    :param invert: also filter for white borders
    ====================================================================================================================
    """
    # black
    l, t, r, b = img.getbbox()
    content = img.crop((l, t, r, b))
    # white
    if invert:
        inv_content = ImageOps.invert(content.copy().convert('RGB'))
        l, t, r, b = inv_content.getbbox()
        content = content.crop((l, t, r, b))
    return content


def fill_frame_promportionate(img, width, height):
    """
    scale proportionally to the fill the full canvas
    ====================================================================================================================
    :param img: source image with alpha
    :param width: output canvas width
    :param height: output canvas height
    ====================================================================================================================
    """
    img_w, img_h = img.size
    delta_w = width - img_w
    delta_h = height - img_h

    # smallest delta needs to get added to both
    if delta_h >= delta_w:
        scale = float(height) / float(img_h)
        new_w = int(round(img_w * scale, 0))
        new_h = int(round(img_h * scale, 0))
    else:
        scale = float(width) / float(img_w)
        new_w = int(round(img_w * scale, 0))
        new_h = int(round(img_h * scale, 0))

    # overscale image so it covers the whole output size
    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    # calc half of the crop for each side
    w = abs((width - new_w)//2)
    h = abs((height - new_h)//2)
    # then crop the bleed
    img = img.crop((w, h, new_w - w, new_h - h))
    return img


def scatter_bg(img,
               width,
               height,
               bg_color ='auto_edge',
               border_thickness = (3, 3, 3, 3),
               auto_border_size = False,
               auto_border_ratio = 10,
               comp_img_edge = True,
               fill_bg = True,
               mask_final = True,
               max_init_it=50,
               max_init_dist = 1000,
               max_opt = 50,
               max_opt_dist = 150,
               blur_bg_amount = 2,
               mask_blur_range = 100,
               mask_bright = 2,
               comp_blur = 2,
               debug = True):
    """
    ====================================================================================================================
    :param img: source image with alpha
    :param width: output canvas width
    :param height: output canvas height
    :param bg_color: tuple to set the bg color, 'auto_edge' for color in edge, 'auto' will detect using the whole image
    :param border_thickness: tuple with sizes for border left, top, right, bottem, to be used with scatter growth
    :param auto_border_size: use 'add_border' as min border when automatically set the border using the auto_border_ratio
    :param auto_border_ratio: devider for auto_border_size, 1 fully fills the border to the image crop
    :param comp_img_edge: fade/choke away some of the image border and comp/mask with the background
    :param fill_bg: make sure the bg is filled after the full comp to prevent alpha from occuring
    :param mask_final: mask forground + generated bg  using the blur_mask_range to the fill color
    :param max_init_it: exponential scattering itteration of pixels to grow/ fill the image with some pixels
    :param max_init_dist: distance from the image/previouse itteration to scatter
    :param max_opt: further grow pixels locally lineaar
    :param max_opt_dist: distance of growth
    :param blur_bg_amount: blur the scattered bg before comping
    :param mask_blur_range: blur amount to mask the generated bg. / final comp, based from the input image
    :param mask_bright: increase / clip the brightness of the mask
    :param comp_blur: amount of blur added when comping scatter noise and median noise
    :param debug: show image process for debugging
    :return: PIL image
    ====================================================================================================================
    """
    if img == None:
        print('no image', img)
        return img
    # get source size
    img_w, img_h = img.size

    # create canvas
    if width != None and height != None:
        bg = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    else:
        bg = Image.new('RGBA', (img_w, img_h), (255, 255, 255, 0))
        width, height = bg.size

    bg_w, bg_h = bg.size

    # center paste content
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    bg.paste(img, offset)
    img = bg
    og_img = img.copy() # copy original


    if bg_color == 'auto':
        bg_color = get_bg_color(img, use_edge = False)

    elif bg_color == 'auto_edge':
        bg_color = get_bg_color(img, use_edge=True)

    elif bg_color != None:
        bg_color = bg_color

    else:
        bg_color = (255, 255, 255, 0)

    if debug:
        Image.new(mode="RGBA", size=(200, 200), color=bg_color).show()

    # calc the auto border size based on content
    l, t, r, b = img.getbbox()
    f = auto_border_ratio # factor to mult with
    if auto_border_size:
        r_delta  = width - r
        b_delta = height - b
        border_size = [l // f, t // f, r_delta // f, b_delta // f]

        # if the size is less then the border given overwrite auto border
        for ind, i in enumerate(border_thickness):
            if border_size [ind] < i:
                border_size [ind] = i
        border_size = tuple(border_size)

    # simple border
    else:
        border_size = border_thickness

    if bg_color:
        # add border
        img = put_color_border(img=img, bg_color=bg_color, size=border_size)
        if debug:
            img.show()

    if (mask_blur_range % 2) == 0:
        mask_blur_range += 1
    # get the edge of the image
    img_spray = get_edge(img = img)

    # initial spread
    for i in range(max_init_it):
        c = get_edge(img = img)
        f = sigmoid_range(pos=i, slope=0.90, x_range=max_init_it, normalize_neg=False)
        # print (f)
        c = c.effect_spread(distance= int(max_init_dist * f))
        img_spray = Image.composite(img_spray, c , img_spray)

    # grow locally
    for i in range(max_opt):
        c = img_spray.copy()
        f = sigmoid_range(pos=i, slope=-0.5, x_range=max_opt, normalize_neg=False)
        c = c.effect_spread(distance= int(max_opt_dist * f))
        img_spray = Image.composite(img_spray, c , img_spray)

    median = img_spray.filter(filter=ImageFilter.MedianFilter(7))
    blurred = img_spray.filter(filter=ImageFilter.GaussianBlur(blur_bg_amount))

    if debug:
        blurred.show()

    # use mask of start image to create a blend mask to blurred bg
    og_mask = make_blurred_mask(img = og_img,
                                mask_blur_range= mask_blur_range,
                                mask_bright=mask_bright,
                                dither_opacity = 0.1,
                                dither_color=bg_color)
    if debug:
        og_mask.show()

    # comp bg
    out_img = Image.composite(img_spray, blurred, og_mask)
    half_mask = set_brightness (img = og_mask, opacity = 0.75)
    out_img = Image.composite(median, out_img, half_mask)
    out_img = out_img.filter(filter=ImageFilter.GaussianBlur(comp_blur))

    # add noise
    dither(img = out_img, opacity=0.1, color=bg_color)

    # comp with original image
    # chook mask a bit
    chook_mask = og_img.split()[-1]
    chook_mask = chook_mask.filter(ImageFilter.MinFilter(9))

    if comp_img_edge:
        for i in range(3):
            chook_mask = chook_mask.effect_spread(distance=1)
        chook_mask = chook_mask.filter(filter=ImageFilter.GaussianBlur(2))
        chook_mask = ImageOps.invert(chook_mask)
        out_img = Image.composite(out_img, og_img, chook_mask)
    else:
        out_img = Image.composite(og_img, out_img, chook_mask)

    if fill_bg :
        out_mask = og_mask if mask_final else out_img
        fill = Image.new('RGBA', (width, height), bg_color)
        out_img = Image.composite(out_img, fill, out_img)
        out_img = Image.composite(out_img, fill, out_mask)
    return out_img

def get_resize_funtions():
    """
    get resize function names
    ====================================================================================================================
    """
    return ["Dont resize",
     "Stretch",
     "Resize and fill bg",
     "Repeat edges",
     "Scatter Fill",
     "Fill Content Proportionatly",
     "Fill Frame Promportionatly",
     "Crop Content"]

def create_folder(path):
    """
    create folder
    ====================================================================================================================
    """
    # if none exists
    if not os.path.exists(path):
        os.makedirs(path)

def save_preset(preset, data):
    """
    save preset
    ====================================================================================================================
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + r"/resize_presets"
    print(path)
    create_folder(path=path)
    fp = os.path.join(path, "{}.pkl".format(preset))
    print ('dada', fp)
    fp = fp.replace(os.sep, '/')
    print('do saving', fp)
    with open(fp, 'wb') as handle:
        # with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_preset(preset):
    """
    load preset
    ====================================================================================================================
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + r"/resize_presets"
    create_folder(path=path)
    fp = os.path.join(path, "{}.pkl".format(preset))
    fp = fp.replace(os.sep, '/')

    if not os.path.exists(fp):
        return
    with open(fp, 'rb') as handle:
        return pickle.load(handle)

def get_user_name():
    """
    load preset
    ====================================================================================================================
    """
    return os.getlogin()

def get_default_presets(preset_name, debug = True):
    preset_dict = dict()

    if preset_name == 'Dont resize' or preset_name == 0:
        preset_dict['debug'] = False

    elif preset_name == 'Stretch' or preset_name == 1:
        preset_dict['debug'] = False

    elif preset_name == 'Resize and fill bg' or preset_name == 2:
        preset_dict ['fill_border_size'] = 0
        preset_dict ['sample'] = 0
        preset_dict['width'] = None
        preset_dict['height'] = None
        preset_dict['bg_color'] = 'auto_edge'
        preset_dict['pre_fill_square'] = 'auto_edge'
        preset_dict['shrink_faded_edges'] = 0
        preset_dict['mask_final'] = False
        preset_dict['mask_blur_range'] = 100
        preset_dict['mask_bright'] = 2
        preset_dict['mask_dither_opacity'] = 0.01
        preset_dict['debug'] = False

    elif preset_name == 'Repeat edges' or preset_name == 3:
        preset_dict ['fill_border_size'] = -1
        preset_dict ['sample'] = 1
        preset_dict['width'] = None
        preset_dict['height'] = None
        preset_dict['bg_color'] = 'auto_edge'
        preset_dict['pre_fill_square'] = 'auto_edge'
        preset_dict['shrink_faded_edges'] = 0
        preset_dict['mask_final'] = True
        preset_dict['mask_blur_range'] = 100
        preset_dict['mask_bright'] = 2
        preset_dict['mask_dither_opacity'] = 0.01
        preset_dict['debug'] = False

    elif preset_name == 'Scatter Fill' or preset_name == 4:
        preset_dict['width'] =None
        preset_dict['height'] = None
        preset_dict['bg_color'] = "auto_edge"
        preset_dict['border_thickness'] = (0, 0, 0, 0)  # natural spread
        preset_dict['auto_border_size'] = False
        preset_dict['auto_border_ratio'] = 20
        preset_dict['comp_img_edge'] = True
        preset_dict['fill_bg'] = True

        # global scatter
        preset_dict['max_init_it'] = 50
        preset_dict['max_init_dist'] = 2000
        # local growth
        preset_dict['max_opt'] = 50
        preset_dict['max_opt_dist'] = 150
        # finalize
        preset_dict['blur_bg_amount'] = 0
        preset_dict['mask_blur_range'] = 0
        preset_dict['mask_final'] = True
        preset_dict['mask_blur_range'] = 100
        preset_dict['mask_bright'] = 2
        preset_dict['comp_blur'] = 1
        preset_dict['debug'] = False

    elif preset_name == 'Fill Content Proportionatly' or preset_name == 5:
        preset_dict['debug'] = False

    elif preset_name == 'Fill Frame Promportionatly' or preset_name == 6:
        preset_dict['debug'] = False

    elif preset_name == 'Crop Content' or preset_name == 7:
        preset_dict['invert'] = True
        preset_dict['debug'] = False

    else:
        raise RuntimeError('default preset not found', preset_name)

    if debug:
        print ('loading default preset',preset_name,  preset_dict)
    return preset_dict





if __name__ == '__main__':
    """
    ====================================================================================================================
    """
    # path = r"C:\Users\trist\Downloads\square.png"
    path = r"C:\Users\trist\Downloads\paint.jpg"
    img = Image.open(path)
    w,h = img.size

    # simple add borders
    add_borders = False
    if add_borders:
        result = add_margin(img = img, margins = (10,20,30,40), color='black')
        result.show()

    # repeat edges
    repeated_edges = False
    if repeated_edges:
        result = repeat_edges(img = img,
                              fill_border_size=None,
                              sample=1,
                              width=w + 500,
                              height=h + 500,
                              bg_color='auto_edge',
                              pre_fill_square='auto_edge',
                              shrink_faded_edges=0,
                              mask_final=True,
                              mask_blur_range=100,
                              mask_bright=2,
                              mask_dither_opacity= 0.01,
                              debug=False
                              )
        result.show()

    # scatter
    scatter = False
    if scatter:
        # if multiple distances, sample bg color before resizing
        bgc = get_bg_color(img=img, use_edge=True)
        for i in range(1,3):
            img = img.resize((w//i, h//i))
            result = scatter_bg(img = img,
                                width=w + 500,
                                height=h + 500,
                                bg_color=bgc,
                                border_thickness= (0, 0, 0, 0),  # natural spread
                                auto_border_size= False,
                                auto_border_ratio= 20,
                                comp_img_edge= True,
                                fill_bg=True,
                                # global scatter
                                max_init_it=50 * (i),
                                max_init_dist=2000 * i,
                                # local growth
                                max_opt=50 * (i),
                                max_opt_dist=150 * i,

                                blur_bg_amount=0,
                                mask_final=True,
                                mask_blur_range=100,
                                mask_bright=2,
                                comp_blur = 1,
                                debug=False)
            result.show()



