from .utils.vector3 import vec3, rgb
from .utils.random import random_in_unit_disk
import numpy as np
from .ray import Ray


class Camera:
    def __init__(
        self,
        look_from,
        look_at,
        screen_width=400,
        screen_height=300,
        field_of_view=90.0,  # degree
        aperture=0.0,
        focal_distance=1.0,
    ):
        self.screen_width = screen_width    # output image width (# of pixels)
        self.screen_height = screen_height  # output image height (# of pixels)
        self.aspect_ratio = float(screen_width) / screen_height

        self.look_from = look_from
        self.look_at = look_at
        self.camera_width = np.tan(field_of_view * np.pi / 180 / 2.0) * 2.0
        self.camera_height = self.camera_width / self.aspect_ratio

        # camera reference basis in world coordinates
        self.cameraFwd = (look_at - look_from).normalize()
        self.cameraRight = (self.cameraFwd.cross(vec3(0.0, 1.0, 0.0))).normalize()
        self.cameraUp = self.cameraRight.cross(self.cameraFwd)

        # if you use a lens_radius >= 0.0 make sure that samples_per_pixel is a large number. Otherwise you'll get a lot of noise
        self.lens_radius = aperture / 2.0
        self.focal_distance = focal_distance

        # Pixels coordinates in camera basis:
        self.x = np.linspace(
            -self.camera_width / 2.0, self.camera_width / 2.0, self.screen_width
        )
        self.y = np.linspace(
            self.camera_height / 2.0, -self.camera_height / 2.0, self.screen_height
        )

        # we are going to cast a total of screen_width * screen_height * samples_per_pixel rays
        # xx,yy store the origin of each ray in a 3d array where the first and second dimension are the x,y coordinates of each pixel
        # and the third dimension is the sample index of each pixel
        xx, yy = np.meshgrid(self.x, self.y)
        self.x = xx.flatten()
        self.y = yy.flatten()

    def get_ray(self, n: vec3) -> Ray:
        """
        Generates rays emitted from the camera position through each pixel on the image plane.
        Each ray casted through each pixel needs to be perturbed slightly to avoid aliasing.

        Args:
        - n: Index of refraction of the scene's participating medium (for air n = 1).

        Returns:
        - A Ray object containing the origin, direction, and other information of the rays.
        """

        # self.x : (1, # of pixels)
        # self.y : (1, # of pixels)
        num_pixel = len(self.x)

        lx, ly = random_in_unit_disk(num_pixel)
        ray_origin = self.look_from + self.lens_radius*(self.cameraRight*lx + self.cameraUp*ly)

        # sensor coordinate with aliasing (~ U(-w/2,w/2))
        # sensor coordinate(z=f) = pixel coordinate(z=1) * f
        x = (self.x + (np.random.rand(num_pixel)-0.5)*(self.camera_width/self.screen_width)) * self.focal_distance
        y = (self.y + (np.random.rand(num_pixel)-0.5)*(self.camera_height/self.screen_height)) * self.focal_distance

        ray_dir = (self.look_from + self.cameraFwd*self.focal_distance + self.cameraRight*x + self.cameraUp*y - ray_origin).normalize()

        # print(ray_origin)
        # print(ray_dir)
        # print(ray_origin.x.shape)
        # print(ray_dir.x.shape)

        return Ray(
            origin=ray_origin,
            dir=ray_dir,
            depth=0,
            n=n,
            reflections=0,
            transmissions=0,
            diffuse_reflections=0,
        )
