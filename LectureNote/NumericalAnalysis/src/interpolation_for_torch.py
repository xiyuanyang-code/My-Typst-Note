import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os


def tensor_to_image(
    tensor, image_path, scale_factor, method_name, output_dir="upsampled_images"
):
    """将 PyTorch 张量转换回图片并保存。"""
    # (1, C, H, W) -> (H, W, C)
    output_np = tensor.squeeze(0).permute(1, 2, 0).numpy()

    # 将 [0, 1] 范围的浮点数转回 [0, 255] 的整数
    output_np = (output_np.clip(0, 1) * 255).astype(np.uint8)

    output_img = Image.fromarray(output_np)

    # 构造保存路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(
        output_dir, f"{base_name}_{method_name}_{scale_factor}x.png"
    )

    output_img.save(save_path)
    print(f"成功保存 {method_name} 结果到: {save_path}")


def upsample_zero_padding(input_tensor, scale_factor):
    """
    通过在现有像素之间插入零值来实现上采样（不使用插值）。

    例如，如果 scale_factor=2，输入 [A, B] 会变为 [A, 0, B, 0]。

    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 (N, C, H, W)。
        scale_factor (int): 上采样的比例因子。

    Returns:
        torch.Tensor: 上采样后的张量。
    """
    if scale_factor <= 1:
        return input_tensor

    N, C, H, W = input_tensor.shape

    # 1. 在 W 维度（宽度）上插入零
    # 目标宽度是 W * scale_factor。需要插入 W * (scale_factor - 1) 列零。
    # 原始张量 [N, C, H, W]

    # 增加一个维度用于保存零，然后重复原始值
    # 例如，如果 scale_factor=2，形状从 (N, C, H, W) -> (N, C, H, W, 1) -> (N, C, H, W, 2)
    temp_tensor = input_tensor.unsqueeze(-1).repeat(1, 1, 1, 1, scale_factor)

    # 现在 temp_tensor 的形状是 (N, C, H, W, scale_factor)。
    # 我们希望在最后一维的每 scale_factor 个元素中，只有第一个是原值，其余是 0。
    # 我们只需要将重复后的 tensor 的第 2 到第 scale_factor 个通道设置为 0
    if scale_factor > 1:
        # 将第 2 到第 scale_factor 个 '副本' 设置为 0。
        # 注意: Python 索引从 0 开始。
        temp_tensor[..., 1:] = 0

    # 将 W 和 scale_factor 合并为一个新的 W' 维度： W' = W * scale_factor
    # (N, C, H, W, scale_factor) -> (N, C, H, W * scale_factor)
    upsampled_w = temp_tensor.reshape(N, C, H, W * scale_factor)

    # 2. 在 H 维度（高度）上插入零
    # 对 upsampled_w 做同样的操作，但作用于 H 维度。
    # 形状 (N, C, H, W') -> (N, C, H, 1, W')
    upsampled_w = upsampled_w.unsqueeze(3).repeat(1, 1, 1, scale_factor, 1)

    # 将第 2 到第 scale_factor 个 '副本' 设置为 0。
    if scale_factor > 1:
        # 作用于 H 维度（索引 3）
        upsampled_w[:, :, 1:, ...] = 0  # 沿着 H 维度的第二个块及其后的块都设置为 0

    # 将 H 和 scale_factor 合并为一个新的 H' 维度： H' = H * scale_factor
    # (N, C, H, scale_factor, W') -> (N, C, H * scale_factor, W')
    final_upsampled_tensor = upsampled_w.reshape(
        N, C, H * scale_factor, W * scale_factor
    )

    return final_upsampled_tensor


def upsample_and_save(image_path, scale_factor, output_dir="images"):
    """
    Reads an image, converts it to a 4D PyTorch tensor (N, C, H, W),
    performs upsampling using Nearest Neighbor and Bilinear interpolation,
    and saves the resulting tensors as images.

    Args:
        image_path (str): The file path to the input image (e.g., 'input.jpg').
        scale_factor (int): The integer multiplier for upsampling the spatial dimensions (H and W).
        output_dir (str, optional): The directory where the upsampled images will be saved.
                                     Defaults to "upsampled_images".

    Returns:
        None: The function saves output files to the specified directory.

    Raises:
        FileNotFoundError: If the specified `image_path` does not exist.
        Exception: For other errors during image processing or tensor operations.

    Notes:
        1. The input image is converted to a normalized float tensor [0, 1].
        2. Nearest Neighbor interpolation results in blocky artifacts but is suitable for
           discrete data like segmentation masks.
        3. Bilinear interpolation results in a smoother image and is generally preferred
           for feature map scaling.
    """
    # 1. Check and create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 2. Read the image and convert to PyTorch Tensor (N, C, H, W)
        img = Image.open(image_path).convert("RGB")

        # PIL Image -> NumPy Array (H, W, C)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # NumPy Array (H, W, C) -> PyTorch Tensor (1, C, H, W)
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        print(f"Original image size: {img.size} (W, H)")
        print(f"Original tensor shape: {input_tensor.shape}")

        # 3. Perform upsampling using different interpolation modes

        # --- A. Nearest Neighbor Interpolation ---
        upsampled_nearest = F.interpolate(
            input_tensor, scale_factor=scale_factor, mode="nearest"
        )

        # --- B. Bilinear Interpolation ---
        upsampled_bilinear = F.interpolate(
            input_tensor,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

        upsampled_zero = upsample_zero_padding(
            input_tensor=input_tensor, scale_factor=scale_factor
        )

        print(f"Upsampled tensor shape: {upsampled_nearest.shape}")

        # 4. Convert Tensor back to Image and save

        def tensor_to_image(tensor, filename, method_name):
            # (1, C, H, W) -> (H, W, C)
            output_np = tensor.squeeze(0).permute(1, 2, 0).numpy()

            # Denormalize [0, 1] to [0, 255] and convert to integer type
            output_np = (output_np.clip(0, 1) * 255).astype(np.uint8)

            output_img = Image.fromarray(output_np)

            # Construct save path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(
                output_dir, f"{base_name}_{method_name}_{scale_factor}x.png"
            )

            output_img.save(save_path)
            print(f"Successfully saved {method_name} result to: {save_path}")

        # Save for zero padding
        tensor_to_image(upsampled_zero, "zero", "Zero")

        # Save Nearest Neighbor result
        tensor_to_image(upsampled_nearest, "nearest", "Nearest")

        # Save Bilinear result
        tensor_to_image(upsampled_bilinear, "bilinear", "Bilinear")

    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
    except Exception as e:
        print(f"An error occurred during image processing: {e}")


if __name__ == "__main__":
    image_file = "./images/Standard.png"
    upsample_factor = 10

    if not os.path.exists(image_file):
        print(f"\n--- Could not find example image '{image_file}' ---")
    else:
        print("--- Starting Upsampling Process ---")
        upsample_and_save(image_file, upsample_factor)
        print("--- Processing Complete ---")
