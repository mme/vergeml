from vergeml.img import ImageType
from vergeml.operation import OperationPlugin, operation
from vergeml.option import option
from PIL import Image

@operation('resize', topic="image", descr="Resize an image to a fixed size.")
@option('width', type=int, descr="Width of the new size.", validate='>0')
@option('height', type=int, descr="Height of the new size.", validate='>0')
@option('channels', type=int, descr="Number of channels.", validate=(0, 3))
@option('mode', type=str, descr="Scaling Mode.", validate=('fill', 'aspect-fill', 'aspect-fit'))
class ResizeOperation(OperationPlugin):
    type = ImageType
    # https://developer.apple.com/documentation/uikit/uiview/contentmode

    def __init__(self, width, height, channels=None, method='antialias', apply=None, mode='aspect-fill'):

        # 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
        # 'nearest': aaaaaaaa|abcd|dddddddd
        # 'reflect': abcddcba|abcd|dcbaabcd
        # 'wrap': abcdabcd|abcd|abcdabcd

        assert(method in ('nearest', 'bicubic', 'antialias', 'bilinear', 'constant', 'nearest', 'reflect', 'warp'))
        assert(mode in ('fill', 'aspect-fill', 'aspect-fit'))
        assert(channels in (None, 1, 3))

        super().__init__(apply)

        self.width = width
        self.height = height
        self.channels = channels
        self.method = method

    def transform(self, img, rng):

        method = getattr(Image, self.method.upper())
        img = img.resize((self.width, self.height), method)
        if self.channels is None:
            return img
        elif self.channels == 1:
            return img.convert('gray')
        elif self.channels == 3:
            return img.convert('RGB')