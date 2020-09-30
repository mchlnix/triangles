from random import randrange
from time import time
from typing import Optional

import cv2
import numpy as np

# how many colors are being considered; if 32 it means only 0, 32, 64, ... , 255 are valid RGB values
COLOR_STEP = 32

# name of the output image
OUTPUT = "output.png"


class Triangler:
    def __init__(self, ref_image: str, start_image: Optional[str] = None):
        self.ref_image = cv2.imread(ref_image)

        if self.ref_image is None:
            raise ValueError(f"Couldn't find input file {ref_image}.")

        if start_image is None:
            self.fit_image = np.zeros(self.ref_image.shape, dtype=np.uint8)
            self.start_image_name = "output.png"
        else:
            self.fit_image = cv2.imread(start_image)
            self.start_image_name = start_image

        self.diffs = np.zeros(0)

        self.height, self.width, *_ = self.ref_image.shape
        self._triangle_length = 0

        self.set_triangles_per_side(10)

        self.color_step = COLOR_STEP

    @property
    def triangle_length(self):
        return self._triangle_length

    @triangle_length.setter
    def triangle_length(self, value):
        self._triangle_length = value

        self.diffs = np.zeros((self.triangle_length, self.triangle_length, 3), dtype=np.uint8)

    def set_triangles_per_side(self, value):
        self.triangle_length = int((self.height + self.width) / 2 / value)
        print(f"Using {self.triangle_length}px triangles.")

    def iterate(self, iterations):
        start = time()

        added = 0

        for i in range(iterations):
            triangle = self.random_triangle()
            triangle = np.array([triangle], dtype=np.int32)

            y, x = triangle[0][0]

            # save a copy on how the area as filled beforehand
            original_area = self.create_patch(x, y)

            # diff the original area, with the changed one
            diff_before = self.diff_of_area(x, y)

            # fill the area with the random triangle
            cv2.fillPoly(self.fit_image, triangle, self.random_color())

            diff_after = self.diff_of_area(x, y)

            if diff_after < diff_before:
                done = i + 1

                elapsed_time = round(time() - start, 2)
                estimated_time = round(elapsed_time/done*iterations, 2)

                print(f"\rDone {done}/{iterations} in {elapsed_time}s/{estimated_time}s", end="")
                added += 1
            elif diff_before < diff_after:
                self.restore_image(original_area, x, y)

        print(f"\rDone in {round(time() - start, 2)} s. Added {added} triangles.")

    def diff_of_area(self, x, y):
        cv2.absdiff(self.fit_image[x: x + self.triangle_length, y: y + self.triangle_length],
                    self.ref_image[x: x + self.triangle_length, y: y + self.triangle_length], self.diffs)

        diff = cv2.sumElems(self.diffs)

        return sum(diff)

    def random_triangle(self):
        x1 = randrange(self.width // self.triangle_length) * self.triangle_length
        x2 = x1 + self.triangle_length - 1

        y1 = randrange(self.height // self.triangle_length) * self.triangle_length
        y2 = y1 + self.triangle_length - 1

        for coordinate in [x1, x2, y1, y2]:
            assert isinstance(coordinate, int)

        p1 = (x1, y1)

        if randrange(2):
            p2 = (x1, y2)
        else:
            p2 = (x2, y1)

        p3 = (x2, y2)

        return [p1, p2, p3]

    def random_color(self):
        r = randrange(256 // self.color_step) * self.color_step
        g = randrange(256 // self.color_step) * self.color_step
        b = randrange(256 // self.color_step) * self.color_step

        return r, g, b

    def restore_image(self, patch, x, y):
        self.fit_image[x: x + self.triangle_length, y: y + self.triangle_length] = patch

    def create_patch(self, x, y):
        return self.fit_image[x: x + self.triangle_length, y: y + self.triangle_length].copy()

    def save(self, image_path=""):
        if not image_path:
            image_path = self.start_image_name

        cv2.imwrite(image_path, self.fit_image)


if __name__ == '__main__':
    # input file
    triangler = Triangler("fit5.jpg")

    # range of the triangles being drawn; if range(2, 4): triangles of size 2 and 3, higher = smaller triangles
    # first larger triangles are drawn, then smaller on top of that
    for i in range(1, 3):
        # set amount of triangles per side (average of width and height)
        triangler.set_triangles_per_side(10 * 2 ** i)

        # amount of time triangles are tried to be added
        triangler.iterate(1_000 * 10 ** i)

        triangler.save(OUTPUT)

    # we want to give the user the option to stop at any time and save the image
