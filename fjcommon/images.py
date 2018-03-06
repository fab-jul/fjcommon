import sys
import glob
import os
import argparse


def _get_PIL_Image():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        print('images.py needs PIL!')
        sys.exit(1)


def _open_image(img_p):
    return _get_PIL_Image().open(img_p)


def central_crop(images_glob, target_w, target_h, append_to_name=''):
    assert isinstance(target_w, int) and isinstance(target_h, int)
    for img_p in _img_ps(images_glob):
        im = _open_image(img_p)
        w, h = im.size
        if w < target_w or h < target_h:
            print('WARN: cannot crop {}, too small!'.format(img_p))
            continue

        left = (w - target_w) // 2
        top = (h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        im_out = im.crop((left, top, right, bottom))

        img_p_base, ext = os.path.splitext(img_p)
        img_p_out = img_p_base + append_to_name + ext
        print('Saving {}...'.format(img_p_out))
        im_out.save(img_p_out)


def resize(images_glob, out_dir, target_short_edge, append_to_name='', new_ext=None, skip_existing=True):
    assert isinstance(target_short_edge, int)
    assert not new_ext or ('.' in new_ext), 'Need . in ext, got {}'.format(new_ext)
    if out_dir is None:
        out_dir = os.path.dirname(images_glob)
    report = []
    for img_p in _img_ps(images_glob):
        img_name, ext = os.path.splitext(os.path.basename(img_p))
        if new_ext:
            ext = new_ext
        img_p_out = os.path.join(out_dir, img_name + append_to_name + ext)

        if os.path.exists(img_p_out):
            if skip_existing:
                print('Already exists: {}, skipping...'.format(img_p_out))
                continue
            print('Already exists, --skip_existing not given. Stopping...')
            return

        print('Resizing {}...'.format(img_p_out))

        im = _open_image(img_p)
        w, h = im.size
        h_is_short_edge = h <= w

        short_edge = h if h_is_short_edge else w
        assert short_edge >= target_short_edge
        long_edge = w if h_is_short_edge else h
        ratio = target_short_edge / short_edge
        new_short = target_short_edge
        new_long = int(long_edge * ratio)

        if h_is_short_edge:
            new_h, new_w = new_short, new_long
        else:
            new_h, new_w = new_long, new_short

        try:
            im_out = im.resize((new_w, new_h), _get_PIL_Image().ANTIALIAS)
            im_out.save(img_p_out)
        except OSError as e:
            print('Caught {}, ignoring'.format(e))
            report.append(e)

    if report:
        print('\n'.join(map(str, report)))


def sizes_of_images_in(images_glob):
    sizes = [_open_image(p).size for p in glob.glob(images_glob)]
    print('{} images, sizes:'.format(len(sizes)))
    print('\n'.join('{}x{}'.format(w, h) for w, h in set(sizes)))


def _img_ps(images_glob):
    img_ps = glob.glob(images_glob)
    if len(img_ps) == 0:
        print('No images matching {}'.format(images_glob))
        return []
    return sorted(img_ps)


def main(args):
    p = argparse.ArgumentParser()
    mode_subparsers = p.add_subparsers(dest='mode', title='Mode')
    # Central Crop ---
    parser_ccrop = mode_subparsers.add_parser('central_crop')
    parser_ccrop.add_argument('imgs_glob', type=str)
    parser_ccrop.add_argument('target_w', type=int)
    parser_ccrop.add_argument('target_h', type=int)
    parser_ccrop.add_argument('--append_name', type=str, default='')
    # Resize ---
    parser_resize = mode_subparsers.add_parser('resize')
    parser_resize.add_argument('imgs_glob', type=str)
    parser_resize.add_argument('target_short_edge', type=int)
    parser_resize.add_argument('--append_name', type=str, default='')
    parser_resize.add_argument('--out_dir', type=str, default=None)
    parser_resize.add_argument('--new_ext', type=str)
    parser_resize.add_argument('--skip_existing', action='store_const', const=True)
    # Sizes ---
    parser_ccrop = mode_subparsers.add_parser('sizes')
    parser_ccrop.add_argument('imgs_glob', type=str)
    flags = p.parse_args(args)
    if flags.mode == 'central_crop':
        central_crop(flags.imgs_glob, flags.target_w, flags.target_h, flags.append_name)
    if flags.mode == 'resize':
        resize(flags.imgs_glob, flags.out_dir, flags.target_short_edge,
               flags.append_name, flags.new_ext, flags.skip_existing)
    elif flags.mode == 'sizes':
        sizes_of_images_in(flags.imgs_glob)
