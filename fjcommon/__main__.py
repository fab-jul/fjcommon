import argparse
from fjcommon import images


def _tf_records():  # loads TF, which is slow
    from fjcommon import tf_records
    return tf_records

def _tf_helpers():  # loads TF, which is slow
    from fjcommon import tf_helpers
    return tf_helpers


files = {
    'tf_records': _tf_records(),
    'tf_helpers': _tf_helpers(),
    'images': images
}

parser = argparse.ArgumentParser()
parser.add_argument('file', choices=files.keys())
flags, remaining_args = parser.parse_known_args()
files[flags.file].main(remaining_args)
