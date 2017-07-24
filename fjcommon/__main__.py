import argparse
from fjcommon import tf_records


files = {
    'tf_records': tf_records
}

parser = argparse.ArgumentParser()
parser.add_argument('file', choices=files.keys())
flags, remaining_args = parser.parse_known_args()
files[flags.file].main(remaining_args)
