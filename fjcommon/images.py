import sys
import glob
import os
import argparse


def _open_image(img_p):
    try:
        from PIL import Image
        return Image.open(img_p)
    except ImportError as e:
        print('images.py needs PIL!')
        sys.exit(1)


def central_crop(images_glob, target_w, target_h, append_to_name=''):
    assert isinstance(target_w, int) and isinstance(target_h, int)
    img_ps = glob.glob(images_glob)
    if len(img_ps) == 0:
        print('No images at {}'.format(images_glob))
        return

    for img_p in sorted(img_ps):
        im = _open_image(img_p)
        w, h = im.size
        if w < target_w or h < target_h:
            print('WARN: cannot crop {}, too small!'.format(img_p))
            continue

        left = (w - target_w) // 2
        top = (h - target_h) // 2
        right = left + target_w
        bottom = top + target_w
        im_out = im.crop((left, top, right, bottom))

        img_p_base, ext = os.path.splitext(img_p)
        img_p_out = img_p_base + append_to_name + ext
        print('Saving {}...'.format(img_p_out))
        im_out.save(img_p_out)


def sizes_of_images_in(images_glob):
    sizes = [_open_image(p).size for p in glob.glob(images_glob)]
    print('{} images, sizes:'.format(len(sizes)))
    print('\n'.join('{}x{}'.format(w, h) for w, h in set(sizes)))


def main(args):
    p = argparse.ArgumentParser()
    mode_subparsers = p.add_subparsers(dest='mode', title='Mode')
    # Central Crop ---
    parser_ccrop = mode_subparsers.add_parser('central_crop')
    parser_ccrop.add_argument('imgs_glob', type=str)
    parser_ccrop.add_argument('target_w', type=int)
    parser_ccrop.add_argument('target_h', type=int)
    parser_ccrop.add_argument('--append_name', type=str, default='')
    # Sizes ---
    parser_ccrop = mode_subparsers.add_parser('sizes')
    parser_ccrop.add_argument('imgs_glob', type=str)
    flags = p.parse_args(args)
    if flags.mode == 'central_crop':
        central_crop(flags.imgs_glob, flags.target_w, flags.target_h, flags.append_name)
    elif flags.mode == 'sizes':
        sizes_of_images_in(flags.imgs_glob)
