# attack.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

def total_variation_loss(img, weight=1e-4):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (N, C, H, W) holding an input image.
    - weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by weight.
    """
    # Your code here: calculate the total variation loss
    N, C, H, W = img.shape
    x_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
    y_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
    loss = weight * (torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))) / N
    return loss

def prototype_inversion_attack(target_model, target_prototype, input_shape, args,
                               num_iterations=1000, lr=0.01, use_tv_loss=True):
    """
    Performs prototype inversion attack.

    Args:
        target_model: The client's model (already loaded and on the correct device).
        target_prototype: The specific local prototype tensor C_i^(j) to invert.
        input_shape: Tuple indicating the shape of the input (e.g., (1, 28, 28)).
        args: Command line arguments (for device).
        num_iterations: Number of optimization steps.
        lr: Learning rate for the optimizer.
        use_tv_loss: Whether to add Total Variation regularization.

    Returns:
        reconstructed_image: The optimized input tensor.
        final_loss: The final L2 distance loss.
    """
    target_model.eval() # Ensure model is in eval mode

    # Initialize a random input tensor (requires gradient)
    # Start with random noise, but ensure it's within a reasonable range (e.g., [0, 1] or normalized)
    # Using uniform noise and then clamping/normalizing can be effective.
    reconstructed_image = torch.randn((1,) + input_shape, device=args.device, requires_grad=True)

    # Optimizer only updates the input image, not the model weights
    optimizer = optim.Adam([reconstructed_image], lr=lr)

    # Ensure target prototype is on the correct device and detached
    target_prototype = target_prototype.clone().detach().to(args.device)

    print(f"Starting prototype inversion attack for {num_iterations} iterations...")
    for i in range(num_iterations):
        optimizer.zero_grad()

        # --- Important: Clamp/Normalize the image ---
        # Clamp to valid image range (e.g., 0-1 if normalized, or based on dataset transforms)
        # This helps prevent unrealistic pixel values. Adjust based on your data normalization.
        with torch.no_grad():
             reconstructed_image.clamp_(0, 1) # Example: clamp if input is expected in [0, 1]
             # Or apply normalization if the model expects normalized input:
             # norm_mean = torch.tensor([0.1307]).view(1,-1,1,1).to(args.device) # MNIST example
             # norm_std = torch.tensor([0.3081]).view(1,-1,1,1).to(args.device)  # MNIST example
             # image_for_model = (reconstructed_image - norm_mean) / norm_std

        # Pass the current reconstructed image through the model's embedding layer
        # The model should return (log_probs, embedding)
        _, current_embedding = target_model(reconstructed_image) # Use reconstructed_image directly if model handles normalization inside

        # Calculate the L2 distance loss between the current embedding and the target prototype
        loss_l2 = F.mse_loss(current_embedding, target_prototype)
        loss = loss_l2

        # Optional: Add Total Variation loss for smoother images
        tv_loss = 0
        if use_tv_loss:
            tv_loss = total_variation_loss(reconstructed_image)
            loss = loss + tv_loss

        # Backpropagate the loss with respect to the image pixels
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Iteration [{i+1}/{num_iterations}], L2 Loss: {loss_l2.item():.4f}, TV Loss: {tv_loss.item() if use_tv_loss else 0:.4f}")

    # Detach the final image from the computation graph
    final_image = reconstructed_image.clone().detach()
    final_loss = F.mse_loss(target_model(final_image)[1], target_prototype).item() # Recalculate final loss

    print(f"Attack finished. Final L2 Loss: {final_loss:.4f}")
    return final_image, final_loss