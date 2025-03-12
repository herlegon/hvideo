from __future__ import annotations
import torch
from torch import Tensor
from torch.types import Number


def pccm_color_transfer(
    x: Tensor,
    ref_x: Tensor,
    valid_indices: Tensor,
    ref_valid_indices: Tensor,
) -> Tensor:
    """
    Transfers the color distribution from the source to the target image using
    Principal Component Color Matching.
    This implementation is based on:
    - Kotera, Hiroaki, Hung-Shing Chen, and Tetsuro Morimoto.
        "Object-to-Object Color Mapping by Image Segmentation." In Color Imaging:
        Device-Independent Color, Color Hardcopy, and Graphic Arts IV, 3648:148-57.
        SPIE, 1998.
    - Kotera, Hiroaki. "A Scene-Referred Color Transfer for Pleasant Imaging
        on Display." In IEEE International Conference on Image Processing 2005,
        2:II-5, 2005.

    https://github.com/dstein64/colortrans
    """
    if x.shape[0] != 1:
        raise ValueError("Batch must be == 1")

    img = x.view(-1, 3).to(dtype=torch.float16).flip(2)
    ref_img = ref_x.view(-1, 3).to(dtype=torch.float16).flip(2)

    shape = img.shape

    # Convert HxWxC image to a (H*W)xC matrix.
    content = img.reshape(-1, shape[-1])
    ref_img = ref_img.reshape(-1, shape[-1])

    valid_content = img[valid_indices]
    valid_reference = ref_img[ref_valid_indices]
    mu_content = torch.mean(valid_content, dim=0)
    mu_reference = torch.mean(valid_reference, dim=0)

    # Calculate covariance matrices
    content_centered = valid_content - mu_content
    reference_centered = valid_reference - mu_reference

    cov_content = torch.matmul(content_centered.T, content_centered) / (valid_content.size(0) - 1)
    cov_reference = torch.matmul(reference_centered.T, reference_centered) / (valid_reference.size(0) - 1)

    # Eigendecomposition
    eigval_content, eigvec_content = torch.linalg.eigh(cov_content)
    eigval_reference, eigvec_reference = torch.linalg.eigh(cov_reference)

    # Division by 0 is forbidden: change null values to arbitrary values
    eigval_content = torch.where(
        eigval_content == 0,
        torch.tensor(1e-42, device=eigval_content.device),
        eigval_content
    )

    eigval_factor = eigval_reference / eigval_content
    scaling = torch.diag(torch.sqrt(torch.clamp(eigval_factor, min=0)))

    transform = torch.matmul(torch.matmul(eigvec_reference, scaling), eigvec_content.T)

    # Apply the transformation
    content_centered = content - mu_content
    transfer = torch.matmul(content_centered, transform.T) + mu_reference

    # Restore image dimensions and clip values
    transfer = transfer.reshape(img.shape).clamp(0, 1)
    transfer = transfer.flip(2).permute(2, 0, 1).unsqueeze(0)

    return transfer.to(dtype=x.dtype)




def matrix_sqrt(x: Tensor) -> Tensor:
    eig_val, eig_vec = torch.linalg.eigh(x)
    return torch.matmul(
        torch.matmul(eig_vec, torch.diag(torch.sqrt(torch.clamp(eig_val, min=0)))),
        eig_vec.T
    )


def lhm_color_transfer(
    x: Tensor,
    ref_x: Tensor,
) -> Tensor:
    """
    Transfers the color distribution from the source to the target image
    using the Linear Histogram Matching.

    This implementation is based on to the Hertzmann, Aaron. "Algorithms
    for Rendering in Artistic Styles." Ph.D., New York University, 2001.

    https://github.com/dstein64/colortrans
    """

    img = x.view(-1, 3).to(dtype=torch.float16).flip(2)
    ref_img = ref_x.view(-1, 3).to(dtype=torch.float16).flip(2)

    shape = img.shape

    # Convert HxWxC image to a (H*W)xC matrix.
    content = img.reshape(-1, shape[-1])

    mu_content = torch.mean(img, dim=0)
    mu_reference = torch.mean(ref_img, dim=0)

    # Calculate covariance matrices
    content_centered = img - mu_content
    reference_centered = ref_img - mu_reference

    cov_content = (
        torch.matmul(content_centered.T, content_centered) / (img.size(0) - 1)
    )
    cov_reference = (
        torch.matmul(reference_centered.T, reference_centered) / (ref_img.size(0) - 1)
    )

    transfer = matrix_sqrt(cov_reference)
    sqrt_cov_content = matrix_sqrt(cov_content)

    # Check if matrix is singular and modify if needed
    if torch.matrix_rank(sqrt_cov_content) < sqrt_cov_content.shape[0]:
        # Singular matrix: modify it by an arbitrary value before calculating the inverse matrix
        sqrt_cov_content += (
            torch.eye(
                sqrt_cov_content.shape[-1],
                device=sqrt_cov_content.device,
                dtype=sqrt_cov_content.dtype
            ) / 255.0
        )

    sqrt_cov_content_inv = torch.linalg.inv(sqrt_cov_content)

    transfer = torch.matmul(transfer, sqrt_cov_content_inv)
    content_centered = content - mu_content
    transfer = torch.matmul(content_centered, transfer.T) + mu_reference

    # Restore image dimensions.
    transfer = transfer.reshape(img.shape).clamp(0, 1)
    transfer = transfer.flip(2).permute(2, 0, 1).unsqueeze(0)

    return transfer.to(dtype=x.dtype)




def stats_color_transfer(source: Tensor, target: Tensor) -> torch.Tensor:
    """
    Computes Reinhard's image colour transfer

    Args:
        source: Source image tensor with shape (H, W, 3) or (B, H, W, 3), values in range [0, 1]
        target: Target image tensor with shape (H, W, 3) or (B, H, W, 3), values in range [0, 1]

    Returns:
        Colour-transferred source image tensor with same shape as source

    References:
        Erik Reinhard, Michael Ashikhmin, Bruce Gooch and Peter Shirley,
        'Color Transfer between Images', IEEE CG&A special issue on Applied
        Perception, Vol 21, No 5, pp 34-41, September - October 2001
    """
    device = source.device

    # Store original shape
    original_shape = source.shape

    # Reshape images to 2D matrices
    img_s = source.reshape(-1, 3)
    img_t = target.reshape(-1, 3)

    # Define transformation matrices
    a = torch.tensor([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ], device=device)

    b = torch.tensor([
        [1/torch.sqrt(torch.tensor(3.0)), 0, 0],
        [0, 1/torch.sqrt(torch.tensor(6.0)), 0],
        [0, 0, 1/torch.sqrt(torch.tensor(2.0))]
    ], device=device)

    c = torch.tensor([
        [1, 1, 1],
        [1, 1, -2],
        [1, -1, 0]
    ], device=device)

    b2 = torch.tensor([
        [torch.sqrt(torch.tensor(3.0))/3, 0, 0],
        [0, torch.sqrt(torch.tensor(6.0))/6, 0],
        [0, 0, torch.sqrt(torch.tensor(2.0))/2]
    ], device=device)

    c2 = torch.tensor([
        [1, 1, 1],
        [1, 1, -1],
        [1, -2, 0]
    ], device=device)

    # Clamp small values to avoid log(0)
    img_s = torch.clamp(img_s, min=1e-8)
    img_t = torch.clamp(img_t, min=1e-8)

    # Convert to LMS space
    LMS_s = torch.matmul(a, img_s.t())
    LMS_t = torch.matmul(a, img_t.t())

    # Take the log of LMS
    LMS_s = torch.log10(LMS_s)
    LMS_t = torch.log10(LMS_t)

    # Convert to lab space
    lab_s = torch.matmul(b, torch.matmul(c, LMS_s))
    lab_t = torch.matmul(b, torch.matmul(c, LMS_t))

    # Compute mean and std
    mean_s = torch.mean(lab_s, dim=1, keepdim=True)
    std_s = torch.std(lab_s, dim=1, keepdim=True)
    mean_t = torch.mean(lab_t, dim=1, keepdim=True)
    std_t = torch.std(lab_t, dim=1, keepdim=True)

    # Statistical alignment for each channel
    sf = std_t / std_s
    res_lab = (lab_s - mean_s) * sf + mean_t

    # Convert back to LMS
    LMS_res = torch.matmul(c2, torch.matmul(b2, res_lab))
    LMS_res = torch.pow(10, LMS_res)

    # Convert back to RGB
    rgb_to_lms_inverse = torch.tensor([
        [4.4679, -3.5873, 0.1193],
        [-1.2186, 2.3809, -0.1624],
        [0.0497, -0.2439, 1.2045]
    ], device=device)

    est_im = torch.matmul(rgb_to_lms_inverse, LMS_res).t()

    # Reshape the image to original dimensions
    est_im = est_im.reshape(original_shape)

    return est_im
