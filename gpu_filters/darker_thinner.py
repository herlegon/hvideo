import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Self


class DarkerThinner(nn.Module):
    def __init__(
        self,
        strength: float = 0.6,
        thin: float = 0.4,
        luma_max: float = 0.75,
        threshold: float = 0.016,
        minimum: float = 0.02,
    ):
        """Make lines darker and thinner
        The input tensor must be in float format (float32 or float16).

        Args:
            - strength: strength used to darken. Disabled when strength is equal to 0.
                    Recommended values: [0.1 .. 2.0]
            - thin: strenght used to thinner lines.
                    High values may create artifacts around the lines.
                    Disabled when strength is equal to 0.
                    Recommended values: [0 .. 0.8]
            - luma_max: do not darken pixels whose Y value is higher that this value.
                    Recommended range: [0.5 .. 1.0]
            - threshold: if the modification of the luma value is less than this this range,
                    a pixel won't be modified. Recommended range: [0.01 .. 0.03]
            - minimum: do not darken lines below this threshold.
                    Recommended range: [0 .. 0.20]
        """
        # Some parts inspired by http://avisynth.nl/index.php/External_filters#Line_Darkening
        super(DarkerThinner, self).__init__()
        self.minimum = minimum
        self.luma_max = luma_max
        self.threshold = threshold
        self.strength = strength
        self.thin = thin

        dtype: torch.dtype = torch.float32

        # YUV <-> RGB
        # Rec.601 (cv2): Y= 0.2126 R + 0.7152 G + 0.0722 B
        # Rec.709: Y= 0.299 R + 0.587 G + 0.114 B
        #
        # R = 1 * Y + 1.28033 * V
        # G = 1 * Y + -0.21482 * U - 0.38059 * V
        # B = 1 * Y + 2.12798 * U

        # BUG wrong order of coefs!
        self.rgb_to_y = torch.tensor(
            [0.114, 0.587, 0.299],
            # [0.0722, 0.7152, 0.2126],
            dtype=dtype,
            requires_grad=False
        )

        self.conv_radius: int = 7
        self.conv_kernel: Tensor = torch.ones(
            [1, 1, self.conv_radius, self.conv_radius],
            dtype=dtype,
            requires_grad=False,
        )
        self.conv_kernel /= (self.conv_radius * self.conv_radius)


    def to(
        self,
        device: str | torch.device | int,
    ) -> Self:
        self.rgb_to_y = self.rgb_to_y.to(device)
        self.conv_kernel = self.conv_kernel.to(device)
        return self


    def half(self) -> Self:
        self.rgb_to_y = self.rgb_to_y.half()
        self.conv_kernel = self.conv_kernel.half()
        return self


    @torch.inference_mode()
    def dilate(
        self,
        tensor: Tensor,
        kernel_size: int = 3,
        iterations: int = 1
    ) -> Tensor:
        max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        for _ in range(iterations):
            tensor = max_pool(tensor)
        return tensor


    @torch.inference_mode()
    def erode(
        self,
        tensor: Tensor,
        kernel_size: int = 3,
        iterations: int = 1
    ) -> Tensor:
        max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        for _ in range(iterations):
            tensor = - max_pool(-tensor)
        return tensor


    @torch.inference_mode()
    def morph_close(
        self,
        tensor: Tensor,
        kernel_size: int | tuple[int, int] = 3,
        iterations: int = 1
    ) -> Tensor:
        max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        for _ in range(iterations):
            tensor = max_pool(tensor)
        for _ in range(iterations):
            tensor = - max_pool(-tensor)
        return tensor


    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        yp: Tensor = torch.tensordot(x, self.rgb_to_y, dims=([1], [0]))
        yp = yp.unsqueeze(0)

        # BUG use bilinear
        h, w = yp.shape[2:]
        yp_2x: Tensor = F.interpolate(
            input=yp,
            size=(h * 2, w * 2),
            mode="bicubic",
            align_corners=False,
            antialias=True
        )
        mask = self.morph_close(yp_2x, kernel_size=3, iterations=3)
        mask: Tensor = F.interpolate(
            input=mask,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
            antialias=True
        )
        mask = torch.where(mask < self.minimum, yp, mask)
        mask_clamped = torch.minimum(
            mask,
            torch.tensor(self.luma_max, dtype=x.dtype)
        )
        mask_lines: Tensor = mask_clamped - yp
        lines: Tensor = torch.where(mask_lines > self.threshold, mask_lines, 0)
        darker_lines = lines * self.strength
        out_yp = torch.clamp(yp - darker_lines, 0, 1)

        if self.thin <= 0:
            yp_diff = out_yp - yp

        else:
            # Reprocess lines
            lines = torch.where(lines > self.threshold, 2 * lines, 0)
            lines_morph: np.ndarray = self.dilate(lines, kernel_size=3, iterations=1)
            lines_morph = 1 - lines_morph * (8 * self.thin)
            lines_morph_conv: Tensor = F.conv2d(
                lines_morph, self.conv_kernel, padding=self.conv_radius // 2
            )
            lines_morph_conv = lines_morph_conv.clamp_(0, 1)

            # Background
            x: Tensor = self.dilate(x, kernel_size=3, iterations=1)
            yp: Tensor = torch.tensordot(x, self.rgb_to_y, dims=([1], [0]))
            yp = yp.unsqueeze(0)
            out_yp = (
                yp * (1 - lines_morph_conv)
                + out_yp * lines_morph_conv
                - darker_lines
            ).clamp_(0, 1)

        yp_diff = out_yp - yp
        yp_diff = yp_diff.expand(*x.shape)
        return x + yp_diff



def darker_thinner(
    img: np.ndarray,
    strength: float = 0.6,
    thin: float = 0.4,
    luma_max: float = 0.75,
    threshold: float = 0.016,
    minimum: float = 0.02,
):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yp = yuv[:, :, 0]

    h, w = yp.shape[:2]
    yp_2x: np.ndarray = cv2.resize(yp, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    mask: np.ndarray = cv2.morphologyEx(
        yp_2x,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=3
    )
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.where(mask < minimum, yp, mask)

    mask_clamped = np.minimum(mask, luma_max)
    mask_lines = mask_clamped - yp
    lines = np.where(mask_lines > threshold, mask_lines, 0)
    darker_lines = lines * strength
    out_yp = np.clip(yp - darker_lines, 0, 1)

    if thin > 0:
        lines = np.where(lines > threshold, 2 * lines, 0)
        lines_morph: np.ndarray = cv2.morphologyEx(
            lines,
            cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1
        )
        lines_morph = 1 - lines_morph * (8 * thin)

        conv_radius: int = 7
        conv_kernel: np.ndarray = (
            np.ones((conv_radius, conv_radius), np.float32)
            / (conv_radius * conv_radius)
        )
        lines_morph_conv = cv2.filter2D(lines_morph, -1, conv_kernel, borderType=cv2.BORDER_CONSTANT)
        lines_morph_conv = np.clip(lines_morph_conv, 0, 1, out=lines_morph_conv)

        # background
        img_d: np.ndarray = cv2.morphologyEx(
            img,
            cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1
        )
        yuv = cv2.cvtColor(img_d, cv2.COLOR_BGR2YUV)
        yp_bgd_d = yuv[:, :, 0]

        out_yp = (
            yp_bgd_d * (1 - lines_morph_conv)
            + out_yp * (lines_morph_conv)
            - darker_lines
        )
        np.clip(out_yp, 0, 1, out=out_yp)

    yuv[:, :, 0] = out_yp
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

