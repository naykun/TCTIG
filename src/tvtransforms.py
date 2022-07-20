import torch
from torch import Tensor
from typing import *
import numbers
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
import warnings
import random
import math

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, sentences):
        for t in self.transforms:
            img, sentences = t(img, sentences)
        return img, sentences
class ToTensor:
    def __init__(self, mean, std, inplace=False, do_norm=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.do_norm = do_norm
    def __call__(self, img, sentences):
        if self.do_norm:
            return F.normalize(F.to_tensor(img), self.mean, self.std, self.inplace), sentences
        else:
            return F.to_tensor(img), sentences

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomChoice:
    """Base class for a list of transformations with randomness
    Args:
        transforms (sequence): list of transformations
    """

    def __init__(self, transforms, weights=None):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms
        self.weights = weights

    def __call__(self, image, sentences):
        t = random.choices(self.transforms, weights=self.weights, k=1)[0]
        return t(image, sentences)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize(torch.nn.Module):
    """Resize the input image to the given size."""
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, sentences):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        # torchvision 0.9.1 do not support antialias
        # return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias), sentences
        return F.resize(img, self.size, self.interpolation), sentences


    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)



class SegmentRandomCrop(torch.nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw
    
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        
    def forward(self, img, sentences):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            sentences: list of dict of caption sentences and their trace

        Returns:
            PIL Image or Tensor: Cropped image.
            sentences: filtered and normalized sentences with corrisponding trace
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        pad_w, pad_h = 0, 0
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            pad_w = self.size[1] - width
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            pad_h = self.size[0] - height
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        
        outimg = F.crop(img, i, j, h, w)
        outsentences = []
        for sentence in sentences:
            processed_sentence = self.process_sentence(sentence, width, height, i, j, h, w, pad_h, pad_w)
            if processed_sentence is not None:
                outsentences.append(processed_sentence)

        return outimg, outsentences
    
    def process_sentence(self, sentence, img_width, img_height, i, j, h, w, pad_h, pad_w, threshold=0.2):
        x_series = [item["x"] for item in sentence["traces"]]
        y_series = [item["y"] for item in sentence["traces"]]
        series = torch.tensor([[item["x"], item["y"], item["t"]] for item in sentence["traces"]], dtype=torch.float).T
        
        x_lower = j / (img_width + 2 * pad_w)
        x_upper = (j + w) / (img_width + 2 * pad_w)
        y_lower = i / (img_height + 2 * pad_h)
        y_upper = (i + h) / (img_height + 2 * pad_h)

        series[0] = (series[0] * img_width + pad_w) / (img_width + 2 * pad_w)
        series[1] = (series[1] * img_height + pad_h) / (img_height + 2 * pad_h)
        
        x_mask = (series[0] > x_lower) & (series[0] < x_upper)
        y_mask = (series[1] > y_lower) & (series[1] < y_upper)
        
        mask = x_mask & y_mask
        exist_rate = mask.sum() / mask.size(0)
        
        if exist_rate > threshold:
            series = series.T[mask]
            series = series.T
            series[0] = (series[0] - x_lower) / (w / (img_width+2*pad_w))
            series[1] = (series[1] - y_lower) / (h / (img_height+2*pad_h))
            series = series.T
            traces = [{"x":item[0], "y":item[1], "t":item[2]} for item in series.tolist()]
            # sentence["aug_traces"] = series
            sentence["traces"] = traces
            return sentence
        else:
            return None

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)
    

class SegmentRandomResizedCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        ratio = [width / height * 0.8 + 0.2] * 2
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, img, sentences):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        pad_h, pad_w = 0, 0
        width, height = F._get_image_size(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        outimg = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        
        outsentences = []
        for sentence in sentences:
            processed_sentence = self.process_sentence(sentence, width, height, i, j, h, w, pad_h, pad_w)
            if processed_sentence is not None:
                outsentences.append(processed_sentence)

        return outimg, outsentences


    def process_sentence(self, sentence, img_width, img_height, i, j, h, w, pad_h, pad_w, threshold=0.2):
        x_series = [item["x"] for item in sentence["traces"]]
        y_series = [item["y"] for item in sentence["traces"]]
        series = torch.tensor([[item["x"], item["y"], item["t"]] for item in sentence["traces"]], dtype=torch.float).T
        
        x_lower = j / (img_width + 2 * pad_w)
        x_upper = (j + w) / (img_width + 2 * pad_w)
        y_lower = i / (img_height + 2 * pad_h)
        y_upper = (i + h) / (img_height + 2 * pad_h)

        series[0] = (series[0] * img_width + pad_w) / (img_width + 2 * pad_w)
        series[1] = (series[1] * img_height + pad_h) / (img_height + 2 * pad_h)
        
        x_mask = (series[0] > x_lower) & (series[0] < x_upper)
        y_mask = (series[1] > y_lower) & (series[1] < y_upper)
        
        mask = x_mask & y_mask
        exist_rate = mask.sum() / mask.size(0)
        
        if exist_rate > threshold:
            series = series.T[mask]
            series = series.T
            series[0] = (series[0] - x_lower) / (w / (img_width+2*pad_w))
            series[1] = (series[1] - y_lower) / (h / (img_height+2*pad_h))
            series = series.T
            traces = [{"x":item[0], "y":item[1], "t":item[2]} for item in series.tolist()]
            # sentence["aug_traces"] = series
            sentence["traces"] = traces
            return sentence
        else:
            return None



    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string    
    
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


if __name__ == "__main__":
    
    img = torch.rand(3,300,300)