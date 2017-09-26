from unittest import TestCase
from . import tf_records


class TestTFRecords(TestCase):
    def test_iterate_in_consecutive_frame_tuples(self):
        ps = [
            'frame0003.png',
            'frame0004.png',
            'frame0005.png',
            'frame0006.png',
            'frame0007.png',
            'frame0008.png',
            'frame0009.png',
        ]
        ps_expected_1 = [
            ['frame0003.png'],
            ['frame0004.png'],
            ['frame0005.png'],
            ['frame0006.png'],
            ['frame0007.png'],
            ['frame0008.png'],
            ['frame0009.png'],
        ]
        ps_expected_3 = [
            ['frame0003.png',
             'frame0004.png',
             'frame0005.png'],
            ['frame0006.png',
             'frame0007.png',
             'frame0008.png'],
        ]
        ps_out_1 = list(tf_records.iterate_in_consecutive_frame_tuples(ps, num_consecutive=1))
        self.assertEqual(ps_out_1, ps_expected_1)
        ps_out_3 = list(tf_records.iterate_in_consecutive_frame_tuples(ps, num_consecutive=3))
        self.assertEqual(ps_out_3, ps_expected_3)

        ps = [
            'vid1_0003.png',
            'vid1_0004.png',
            'vid1_0005.png',
            'vid1_0006.png',
            'vid2_0007.png',
            'vid2_0008.png',
            'vid2_0009.png',
            'vid2_0010.png',
            'vid2_0011.png',
        ]
        ps_expected = [
            ['vid1_0003.png',
             'vid1_0004.png',
             'vid1_0005.png'],
            ['vid2_0009.png',
             'vid2_0010.png',
             'vid2_0011.png'],
        ]
        ps_out = list(tf_records.iterate_in_consecutive_frame_tuples(ps, num_consecutive=3))
        self.assertEqual(ps_out, ps_expected)

        ps_wrong = ['asdf.png']
        with self.assertRaises(ValueError):
            list(tf_records.iterate_in_consecutive_frame_tuples(ps_wrong, num_consecutive=1))

    def test_wrap_frames_in_feature_dicts(self):
        ps = [
            ['vid1_0003.png',
             'vid1_0004.png',
             'vid1_0005.png'],
            ['vid2_0009.png',
             'vid2_0010.png',
             'vid2_0011.png'],
        ]
        out = list(tf_records.wrap_frames_in_feature_dicts(ps, 'K'))
        print('hi')
        print(out)

        ps = [
            ['frame0003.png'],
            ['frame0004.png'],
            ['frame0005.png'],
            ['frame0006.png'],
            ['frame0007.png'],
            ['frame0008.png'],
            ['frame0009.png'],
        ]
        out = list(tf_records.wrap_frames_in_feature_dicts(ps, 'K'))
        print('hi')
        print(out)


def foo():
    pass
