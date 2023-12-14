import random
import cv2
import numpy as np

def convert_keypoints(sample):
    label = sample['label']
    h, w, _ = sample['image'].shape
    keypoints = label['keypoints']

    for keypoint in keypoints:
        if keypoint[0] == keypoint[1] == 0:
            keypoint[2] = 2
        if keypoint[0] < 0 or keypoint[0] >= w or keypoint[1] < 0 or keypoint[1] >= h:
            keypoint[2] = 2

    for other_label in label['processed_other_annotations']:
        keypoints = other_label['keypoints']
        for keypoint in keypoints:
            if keypoint[0] == keypoint[1] == 0:
                keypoint[2] = 2
            if keypoint[0] < 0 or keypoint[0] >= w or keypoint[1] < 0 or keypoint[1] >= h:
                keypoint[2] = 2

    label['keypoints'] = convert_keypoints_order(label['keypoints'], w, h)

    for other_label in label['processed_other_annotations']:
        other_label['keypoints'] = convert_keypoints_order(other_label['keypoints'], w, h)

    return sample

def convert_keypoints_order(keypoints, w, h):
    reorder_map = [1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    converted_keypoints = list(keypoints[i - 1] for i in reorder_map)
    converted_keypoints.insert(1, [(keypoints[5][0] + keypoints[6][0]) / 2,
                                   (keypoints[5][1] + keypoints[6][1]) / 2, 0])

    if keypoints[5][2] == 2 or keypoints[6][2] == 2:
        converted_keypoints[1][2] = 2
    elif keypoints[5][2] == 1 and keypoints[6][2] == 1:
        converted_keypoints[1][2] = 1

    if (converted_keypoints[1][0] < 0 or converted_keypoints[1][0] >= w
            or converted_keypoints[1][1] < 0 or converted_keypoints[1][1] >= h):
        converted_keypoints[1][2] = 2

    return converted_keypoints

class ImageProcessor:
    def __init__(self):
        self.scale_prob = 1
        self.min_scale = 0.5
        self.max_scale = 1.1
        self.target_dist = 0.6

    def scale(self, sample):
        prob = random.random()
        scale_multiplier = 1

        if prob <= self.scale_prob:
            prob = random.random()
            scale_multiplier = (self.max_scale - self.min_scale) * prob + self.min_scale

        label = sample['label']
        scale_abs = self.target_dist / label['scale_provided']
        scale = scale_abs * scale_multiplier

        sample['image'] = cv2.resize(sample['image'], dsize=(0, 0), fx=scale, fy=scale)
        label['img_height'], label['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.resize(sample['mask'], dsize=(0, 0), fx=scale, fy=scale)

        label['objpos'][0] *= scale
        label['objpos'][1] *= scale

        for keypoint in sample['label']['keypoints']:
            keypoint[0] *= scale
            keypoint[1] *= scale

        for other_annotation in sample['label']['processed_other_annotations']:
            other_annotation['objpos'][0] *= scale
            other_annotation['objpos'][1] *= scale

            for keypoint in other_annotation['keypoints']:
                keypoint[0] *= scale
                keypoint[1] *= scale

        return sample

class ImageAugmentor:
    def __init__(self):
        self.pad = (0, 0, 0)
        self.max_rotate_degree = 40

    def rotate(self, sample):
        prob = random.random()
        degree = (prob - 0.5) * 2 * self.max_rotate_degree
        h, w, _ = sample['image'].shape
        img_center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(img_center, degree, 1)

        abs_cos = abs(R[0, 0])
        abs_sin = abs(R[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        dsize = (bound_w, bound_h)

        R[0, 2] += dsize[0] / 2 - img_center[0]
        R[1, 2] += dsize[1] / 2 - img_center[1]

        sample['image'] = cv2.warpAffine(sample['image'], R, dsize=dsize,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self.pad)
        sample['label']['img_height'], sample['label']['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.warpAffine(sample['mask'], R, dsize=dsize,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))

        label = sample['label']
        label['objpos'] = self._rotate(label['objpos'], R)

        for keypoint in label['keypoints']:
            point = [keypoint[0], keypoint[1]]
            point = self._rotate(point, R)
            keypoint[0], keypoint[1] = point[0], point[1]

        for other_annotation in label['processed_other_annotations']:
            for keypoint in other_annotation['keypoints']:
                point = [keypoint[0], keypoint[1]]
                point = self._rotate(point, R)
                keypoint[0], keypoint[1] = point[0], point[1]

        return sample

    def _rotate(self, point, R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]


class KeypointConverter:
    def __call__(self, sample):
        label = sample['label']
        h, w, _ = sample['image'].shape
        keypoints = label['keypoints']

        for keypoint in keypoints:
            if keypoint[0] == keypoint[1] == 0:
                keypoint[2] = 2

            if (
                keypoint[0] < 0
                or keypoint[0] >= w
                or keypoint[1] < 0
                or keypoint[1] >= h
            ):
                keypoint[2] = 2

        for other_label in label['processed_other_annotations']:
            keypoints = other_label['keypoints']

            for keypoint in keypoints:
                if keypoint[0] == keypoint[1] == 0:
                    keypoint[2] = 2

                if (
                    keypoint[0] < 0
                    or keypoint[0] >= w
                    or keypoint[1] < 0
                    or keypoint[1] >= h
                ):
                    keypoint[2] = 2

        label['keypoints'] = self._convert(label['keypoints'], w, h)

        for other_label in label['processed_other_annotations']:
            other_label['keypoints'] = self._convert(
                other_label['keypoints'], w, h
            )

        return sample

    def _convert(self, keypoints, w, h):
        reorder_map = [1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        converted_keypoints = list(keypoints[i - 1] for i in reorder_map)
        converted_keypoints.insert(
            1,
            [
                (keypoints[5][0] + keypoints[6][0]) / 2,
                (keypoints[5][1] + keypoints[6][1]) / 2,
                0,
            ],
        )

        if keypoints[5][2] == 2 or keypoints[6][2] == 2:
            converted_keypoints[1][2] = 2
        elif keypoints[5][2] == 1 and keypoints[6][2] == 1:
            converted_keypoints[1][2] = 1

        if (
            converted_keypoints[1][0] < 0
            or converted_keypoints[1][0] >= w
            or converted_keypoints[1][1] < 0
            or converted_keypoints[1][1] >= h
        ):
            converted_keypoints[1][2] = 2

        return converted_keypoints

class Scaler:
    def __init__(self, prob=1, min_scale=0.5, max_scale=1.1, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_dist = target_dist

    def __call__(self, sample):
        prob = random.random()
        scale_multiplier = 1

        if prob <= self._prob:
            prob = random.random()
            scale_multiplier = (
                (self._max_scale - self._min_scale) * prob + self._min_scale
            )

        label = sample['label']
        scale_abs = self._target_dist / label['scale_provided']
        scale = scale_abs * scale_multiplier
        sample['image'] = cv2.resize(
            sample['image'], dsize=(0, 0), fx=scale, fy=scale
        )
        label['img_height'], label['img_width'], _ = sample['image'].shape
        sample['mask'] = cv2.resize(
            sample['mask'], dsize=(0, 0), fx=scale, fy=scale
        )

        label['objpos'][0] *= scale
        label['objpos'][1] *= scale

        for keypoint in sample['label']['keypoints']:
            keypoint[0] *= scale
            keypoint[1] *= scale

        for other_annotation in sample['label']['processed_other_annotations']:
            other_annotation['objpos'][0] *= scale
            other_annotation['objpos'][1] *= scale

            for keypoint in other_annotation['keypoints']:
                keypoint[0] *= scale
                keypoint[1] *= scale

        return sample


