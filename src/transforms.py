# -*- coding: utf-8 -*-

# Libraries
import numpy as np
import cv2
import pywt
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter
from typing import Dict, List, Optional, Tuple
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
import torch
from torch import Tensor
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple


def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class AugMix(torch.nn.Module):
    r"""AugMix data augmentation method based on
    `"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" <https://arxiv.org/abs/1912.02781>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        severity (int): The severity of base augmentation operators. Default is ``3``.
        mixture_width (int): The number of augmentation chains. Default is ``3``.
        chain_depth (int): The depth of augmentation chains. A negative value denotes stochastic depth sampled from the interval [1, 3].
            Default is ``-1``.
        alpha (float): The hyperparameter for the probability distributions. Default is ``1.0``.
        all_ops (bool): Use all operations (including brightness, contrast, color and sharpness). Default is ``True``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        s = {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if self.all_ops:
            s.update(
                {
                    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                }
            )
        return s

    @torch.jit.unused
    def _pil_to_tensor(self, img) -> Tensor:
        return F.pil_to_tensor(img)

    @torch.jit.unused
    def _tensor_to_pil(self, img: Tensor):
        return F.to_pil_image(img)

    def _sample_dirichlet(self, params: Tensor) -> Tensor:
        # Must be on a separate method so that we can overwrite it in tests.
        return torch._sample_dirichlet(params)

    def forward(self, orig_img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, (height, width) = F.get_image_num_channels(orig_img), F.get_image_size(orig_img)
        if isinstance(orig_img, Tensor):
            img = orig_img
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]
        else:
            img = self._pil_to_tensor(orig_img)

        op_meta = self._augmentation_space(self._PARAMETER_MAX, (height, width))

        orig_dims = list(img.shape)
        batch = img.view([1] * max(4 - img.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)

        # Sample the beta weights for combining the original and augmented image. To get Beta, we use a Dirichlet
        # with 2 parameters. The 1st column stores the weights of the original and the 2nd the ones of augmented image.
        m = self._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1)
        )

        # Sample the mixing weights and combine them with the ones sampled from Beta for the augmented images.
        combined_weights = self._sample_dirichlet(
            torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)
        ) * m[:, 1].view([batch_dims[0], -1])

        mix = m[:, 0].view(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[torch.randint(self.severity, (1,), dtype=torch.long)].item())
                    if magnitudes.ndim > 0
                    else 0.0
                )
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                aug = _apply_op(aug, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            mix.add_(combined_weights[:, i].view(batch_dims) * aug)
        mix = mix.view(orig_dims).to(dtype=img.dtype)

        if not isinstance(orig_img, Tensor):
            return self._tensor_to_pil(mix)
        return mix


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"severity={self.severity}"
            f", mixture_width={self.mixture_width}"
            f", chain_depth={self.chain_depth}"
            f", alpha={self.alpha}"
            f", all_ops={self.all_ops}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


def NormalizeData(data):
    """helper to normalize in [0,1] for the plots"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def apply_noise_and_blur(image, sigma_n, sigma_b):

    blurred = image.filter(ImageFilter.GaussianBlur(radius = sigma_b))
    # Convert the PIL image to a NumPy array
    image_array = np.array(blurred)
    
    # Apply random Gaussian noise
    noise = np.random.normal(scale=sigma_n, size=image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    
    result_image = Image.fromarray(np.uint8(noisy_image))
    
    return result_image

def wavelet_transform_and_reconstruct(image, num_levels, coefficient_reduction_prob):
    # Convert the PIL image to a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Compute the 2D dyadic wavelet transform
    coeffs = pywt.wavedec2(image, wavelet='haar', level=num_levels)
    
    # Iterate through each coefficient level (except the approximation coefficients)
    for level in range(1, num_levels + 1):
        # Get the detail coefficients at the current level
        coeff_level = coeffs[num_levels - level]
        
        # Apply random coefficient reduction to each coefficient type (cH, cV, cD)
        for i in range(3):
            coeff_level[i][np.random.rand(*coeff_level[i].shape) < coefficient_reduction_prob] = 0
        
        # Assign the modified detail coefficients back to the coefficients tuple
        coeffs[num_levels - level] = coeff_level
    
    # for i in range(3): # set the outermost coefficients to 0
    #    coeffs[-1][i][:] = 0

    
    # Reconstruct the altered image using the modified coefficients
    reconstructed_image = pywt.waverec2(coeffs, wavelet='haar')

    image = (NormalizeData(reconstructed_image) * 255).astype(np.uint8)
    
    # Clip values to ensure they are in the valid range for image pixels (0 to 255)
    # reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    
    return Image.fromarray(image)

class Spectral(object):
    """Sets the outermost coefficients of the dyadic WT of the input image
    to 0.
    for the remaining detail coefficients, randomly masks them to force the model to rely on
    multiple components and scale at once
    only keeps the approximation coefficnents unchanged

    Args:
        levels (int): Desired number of levels for the multilevel 
        wavelet transform.
    """

    def __init__(self, levels = 5, coefficient_reduction_prob = 0.2, sigma_b = 1.):
        self.levels = levels
        self.coefficient_reduction_prob = coefficient_reduction_prob
        self.sigma_b = sigma_b

    def __call__(self, sample):

        # apply a blur first

        blurred_image = np.array(apply_noise_and_blur(sample, 0., self.sigma_b))



        out = np.zeros(blurred_image.shape)

        for c in range(out.shape[2]):
            
            image = wavelet_transform_and_reconstruct(blurred_image[:,:,c], self.levels, self.coefficient_reduction_prob)

            out[:,:,c] = NormalizeData(image) * 255
        
        return Image.fromarray(out.astype(np.uint8))

class NoiseAndBlur(object):
    """
    Applies random gaussian noise and gaussian blur 
    with values ranging between 0 and sigma_b for the blur
    and 0 and sigma_n for the noise

    args:
        sigma_b (float) the maximum level of blur
        sigma_n (float) the maximum level of noise
    """

    def __init__(self, sigma_b = 3, sigma_n = 50, deterministic = False):
        self.sigma_b = sigma_b
        self.sigma_n = sigma_n
        self.deterministic = deterministic

    def __call__(self, sample):

        # pick values between 0 and sigma for both blur levels
        if not self.deterministic:
            sn = np.random.choice(np.linspace(0, self.sigma_n, 50), 1).item()
            sb = np.random.choice(np.linspace(0, self.sigma_b, 50), 1).item()

        else: 
            sn = self.sigma_n
            sb = self.sigma_b

        image = apply_noise_and_blur(sample, sn, sb)

        return image