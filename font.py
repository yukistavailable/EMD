import os

import numpy as np

from PIL import Image, ImageFont, ImageDraw
import json
import collections

from torch import nn
from torchvision import transforms

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk.json"
JP_CHARSET_PATH = "./charset/jp.json"


def load_charset():
    cjk = json.load(open(DEFAULT_CHARSET))
    jp = json.load(open(JP_CHARSET_PATH))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = jp["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]

    return {
        'CN': CN_CHARSET,
        'JP': JP_CHARSET,
        'KR': KR_CHARSET,
        'CN_T': CN_T_CHARSET,
    }


def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)

    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    hash値が被っている文字があればそれは欠陥文字であるとみなす。（例えば空白になっている、draw_single_charの返り値がNoneになっているなど）
    そのような文字を取り除く。
    """
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        if img is None:
            hash_count[0] += 1
        else:
            hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def draw_font2font_example(
        ch,
        src_font,
        dst_font,
        canvas_size,
        x_offset,
        y_offset,
        filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new(
        "RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


def draw_fonts_example(
        ch,
        fonts,
        canvas_size,
        char_size,
        x_offset,
        y_offset,
        filter_hashes):
    imgs = []
    for font in fonts:
        _font = ImageFont.truetype(font, size=char_size)
        img = draw_single_char(ch, _font, canvas_size, x_offset, y_offset)
        if img is None:
            _hash = 0
        else:
            _hash = hash(img.tobytes())
        if _hash in filter_hashes or img is None:
            return None
        imgs.append(img)

    example_img = Image.new(
        "RGB", (canvas_size * len(fonts), canvas_size), (255, 255, 255))
    for i, img in enumerate(imgs):
        example_img.paste(img, (canvas_size * i, 0))

    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


def font2font(
        fonts,
        charset,
        char_size=256,
        canvas_size=256,
        x_offset=0,
        y_offset=0,
        sample_count=1000,
        sample_dir='dir',
        label=0,
        filter_by_hash=True):

    filter_hashes = set()
    if filter_by_hash:
        for font in fonts:
            _font = ImageFont.truetype(font, size=char_size)
            print(font)
            _filter_hashes = set(
                filter_recurring_hash(
                    charset,
                    _font,
                    canvas_size,
                    x_offset,
                    y_offset))
            filter_hashes = filter_hashes | _filter_hashes
            print("filter hashes -> %s" %
                  (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_fonts_example(
            c,
            fonts,
            canvas_size,
            char_size,
            x_offset,
            y_offset,
            filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


def draw_same_font_example(
        cs,
        font,
        canvas_size,
        char_size,
        x_offset,
        y_offset,
        filter_hashes):
    font = ImageFont.truetype(font, size=char_size)
    imgs = []
    for ch in cs:
        img = draw_single_char(ch, font, canvas_size, x_offset, y_offset)

        if img is None:
            _hash = 0
        else:
            _hash = hash(img.tobytes())

        if _hash in filter_hashes or img is None:
            return None

        imgs.append(img)

    example_img = Image.new(
        "RGB", (canvas_size * len(cs), canvas_size), (255, 255, 255))

    for i, img in enumerate(imgs):
        example_img.paste(img, (canvas_size * i, 0))

    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


def same_font(
        font,
        charset,
        char_num=5,
        char_size=256,
        canvas_size=256,
        x_offset=0,
        y_offset=0,
        sample_dir='style_dir',
        label=0,
        filter_by_hash=True):

    filter_hashes = set()
    if filter_by_hash:
        _font = ImageFont.truetype(font, size=char_size)
        print(font)
        filter_hashes = set(
            filter_recurring_hash(
                charset,
                _font,
                canvas_size,
                x_offset,
                y_offset))
        print("filter hashes -> %s" %
              (",".join([str(h) for h in filter_hashes])))

    count = 0

    for i in range(0, len(charset), char_num):
        cs = charset[i:i + char_num]
        if len(cs) < char_num:
            return
        e = draw_same_font_example(
            cs,
            font,
            canvas_size,
            char_size,
            x_offset,
            y_offset,
            filter_hashes
        )
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


def generate_paired_data(
        fonts=None,
        charset='JP',
        sample_dir='dir',
        sample_count=2500):
    assert charset in ['CN', 'JP', 'KR', 'CN_T']
    charset = load_charset().get(charset)

    if fonts is None:
        fonts = [
            'fonts/JP_Ronde-B_square.otf',
            'fonts/JP_NotoSerifJP-Regular.otf',
            'fonts/JP_ReggaeOne-Regular.ttf',
            'fonts/JP_saruji.ttf',
            'fonts/JP_TsunagiGothic.ttf',
            'fonts/JP_yokomoji.otf',
        ]

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    font2font(fonts, charset, sample_dir=sample_dir, sample_count=sample_count)


def generate_same_font_data(
        font=None,
        charset='JP',
        char_num=5,
        sample_dir='style_dir',
        shuffle=True,
):
    assert charset in ['CN', 'JP', 'KR', 'CN_T']
    charset = load_charset().get(charset)

    if font is None:
        font = 'fonts/JP_Ronde-B_square.otf'

    if shuffle:
        np.random.shuffle(charset)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    same_font(font, charset, char_num=char_num, sample_dir=sample_dir)
