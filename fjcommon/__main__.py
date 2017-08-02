import argparse
from fjcommon import images


def _tf_records():  # loads TF, which is slow
    from fjcommon import tf_records
    return tf_records


files = {
    'tf_records': _tf_records(),
    'images': images
}

parser = argparse.ArgumentParser()
parser.add_argument('file', choices=files.keys())
flags, remaining_args = parser.parse_known_args()
files[flags.file].main(remaining_args)
