import os
import torch 
import torch.nn.functional as F

def image_to_binary_mask(image: torch.Tensor) -> torch.Tensor:
    image = image.squeeze(0) 
    channel_0 = (image == 0).float()  # Channel for pixels with value 0
    channel_1 = (image == 255).float()  # Channel for pixels with value 255

    binary_mask = torch.stack([channel_0, channel_1], dim=0)
    return binary_mask

def binary_mask_to_image(binary_mask: torch.Tensor) -> torch.Tensor:
    # Channel 0 contributes value 0, Channel 1 contributes value 255
    reconstructed_image = binary_mask[1] * 255  
    return reconstructed_image.unsqueeze(0)  

def next_power_of_2(x: int) -> int:
    """Return the next power of 2 greater than or equal to x."""
    return 1 << (x - 1).bit_length()

def pad_right_bottom(x: torch.Tensor) -> torch.Tensor:
    _, h, w = x.shape 
    
    new_h = next_power_of_2(h)
    new_w = next_power_of_2(w)
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    padded_x = F.pad(x, (0, pad_w, 0, pad_h))  # Padding (left, right, top, bottom)
    return padded_x.to(dtype=x.dtype)

def crop_from_right_bottom(x: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
    _, h, w = x.shape  
    out_h, out_w = output_shape[1], output_shape[2]
    
    cropped_x = x[:, :out_h, :out_w]
    return cropped_x


def save_checkpoint(state: dict, dir_path: str, filename: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, filename)
    torch.save(state, save_path)
    print(f"Checkpoint saved at: {save_path}")

def load_checkpoint(dir_path: str, filename: str) -> dict:
    load_path = os.path.join(dir_path, filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")
    print(f"Checkpoint loaded from: {load_path}")
    return torch.load(load_path)

def uint8_to_float32(image: torch.Tensor) -> torch.Tensor:
    assert image.dtype == torch.uint8, "Input tensor must be of dtype uint8."
    return image.to(dtype=torch.float32) / 255.0

def float32_to_uint8(image: torch.Tensor) -> torch.Tensor:
    assert image.dtype == torch.float32, "Input tensor must be of dtype float32."
    return (image.clamp(0, 1) * 255).to(dtype=torch.uint8)


def softmax_to_binary_masks(softmax_output: torch.Tensor) -> torch.Tensor:
    # Compute binary mask for each class
    class_indices = torch.argmax(softmax_output, dim=0)  # Shape: (H, W)
    
    # Initialize a tensor of zeros for the class masks
    masks = torch.zeros_like(softmax_output, dtype=torch.uint8)  # Shape: (2, H, W)
    
    # Fill masks based on class indices
    masks[0] = (class_indices == 0).to(torch.uint8)  # Mask for class 0
    masks[1] = (class_indices == 1).to(torch.uint8)  # Mask for class 1

    return masks