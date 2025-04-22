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
    # Ensure indices are valid for subtraction
    if H > 1:
        x_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
    else:
        x_diff = torch.zeros_like(img) # Handle single row images

    if W > 1:
        y_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
    else:
        y_diff = torch.zeros_like(img) # Handle single column images

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
        args: Command line arguments (for device, dataset info).
        num_iterations: Number of optimization steps.
        lr: Learning rate for the optimizer.
        use_tv_loss: Whether to add Total Variation regularization.

    Returns:
        reconstructed_image: The optimized input tensor.
        final_loss: The final L2 distance loss.
    """
    target_model.eval() # Ensure model is in eval mode

    # Initialize a random input tensor (requires gradient)
    reconstructed_image = torch.randn((1,) + input_shape, device=args.device, requires_grad=True)

    # Optimizer only updates the input image, not the model weights
    optimizer = optim.Adam([reconstructed_image], lr=lr)

    # Ensure target prototype is on the correct device and detached
    target_prototype = target_prototype.clone().detach().to(args.device)

    print(f"Starting prototype inversion attack for {num_iterations} iterations (lr={lr}, TV={use_tv_loss})...")

    # Determine normalization parameters based on dataset
    if args.dataset == 'mnist':
        norm_mean = torch.tensor([0.1307]).view(1,-1,1,1).to(args.device)
        norm_std = torch.tensor([0.3081]).view(1,-1,1,1).to(args.device)
        clamp_min, clamp_max = -norm_mean[0,0,0,0]/norm_std[0,0,0,0], (1-norm_mean[0,0,0,0])/norm_std[0,0,0,0] # Approx range after normalization
    elif args.dataset == 'femnist':
         # FEMNIST uses same normalization as MNIST in this codebase
        norm_mean = torch.tensor([0.1307]).view(1,-1,1,1).to(args.device)
        norm_std = torch.tensor([0.3081]).view(1,-1,1,1).to(args.device)
        clamp_min, clamp_max = -norm_mean[0,0,0,0]/norm_std[0,0,0,0], (1-norm_mean[0,0,0,0])/norm_std[0,0,0,0]
    elif args.dataset == 'cifar10':
        # Values from trans_cifar10_train
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).to(args.device)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).to(args.device)
        # Clamping for normalized data is less intuitive, maybe clamp to a few std devs?
        # For simplicity, let's clamp the *denormalized* version to [0, 1] conceptually.
        # We'll apply normalization before model input but clamp raw values.
        clamp_min, clamp_max = 0, 1 # Clamping the conceptual unnormalized image
    elif args.dataset == 'cifar100':
        # Values from trans_cifar100_train
        norm_mean = torch.tensor([0.507, 0.487, 0.441]).view(1,-1,1,1).to(args.device)
        norm_std = torch.tensor([0.267, 0.256, 0.276]).view(1,-1,1,1).to(args.device)
        clamp_min, clamp_max = 0, 1 # Clamping the conceptual unnormalized image
    else:
        # Default or error
        print("Warning: Unknown dataset for normalization/clamping in attack. Using default clamp [0, 1].")
        norm_mean, norm_std = None, None
        clamp_min, clamp_max = 0, 1

    best_loss = float('inf')
    best_image = None

    for i in range(num_iterations):
        optimizer.zero_grad()

        # --- Image processing ---
        with torch.no_grad():
            # Conceptually clamp to [0, 1] for unnormalized image, then normalize if needed
            if norm_mean is not None and norm_std is not None:
                # Denormalize, clamp, renormalize
                img_denorm = reconstructed_image * norm_std + norm_mean
                img_denorm.clamp_(clamp_min, clamp_max)
                image_for_model = (img_denorm - norm_mean) / norm_std
                # Update reconstructed_image with the clamped+renormalized version for TV loss
                reconstructed_image.data = image_for_model.data
            else:
                # If no normalization, just clamp
                reconstructed_image.clamp_(clamp_min, clamp_max)
                image_for_model = reconstructed_image


        # Pass the current reconstructed image through the model's embedding layer
        _, current_embedding = target_model(image_for_model)

        # Calculate the L2 distance loss between the current embedding and the target prototype
        loss_l2 = F.mse_loss(current_embedding, target_prototype)
        loss = loss_l2

        # Optional: Add Total Variation loss for smoother images
        tv_loss_val = 0
        if use_tv_loss:
            tv_loss = total_variation_loss(reconstructed_image, weight=1e-4) # Adjust weight as needed
            loss = loss + tv_loss
            tv_loss_val = tv_loss.item()

        # Backpropagate the loss with respect to the image pixels
        loss.backward()
        optimizer.step()

        # Keep track of the best image found so far
        if loss_l2.item() < best_loss:
            best_loss = loss_l2.item()
            best_image = reconstructed_image.clone().detach()

        if (i + 1) % 100 == 0:
            print(f"Iteration [{i+1}/{num_iterations}], L2 Loss: {loss_l2.item():.4f}, TV Loss: {tv_loss_val:.4f}")

    # Denormalize the final best image for visualization if needed
    if norm_mean is not None and norm_std is not None and best_image is not None:
        final_image = best_image * norm_std + norm_mean
        final_image.clamp_(0, 1) # Clamp final result to [0, 1] for visualization
    elif best_image is not None:
         final_image = best_image.clamp_(0, 1) # Clamp if no normalization
    else:
         # Fallback if no iterations improved over initial state
         final_image = torch.zeros((1,) + input_shape, device=args.device)
         print("Warning: No improvement found during attack iterations.")


    # Recalculate final loss with the best image found
    final_loss_val = float('inf')
    if best_image is not None:
        with torch.no_grad():
            _, final_embedding = target_model(best_image) # Pass the normalized best_image
            final_loss_val = F.mse_loss(final_embedding, target_prototype).item()

    print(f"Attack finished. Best L2 Loss: {final_loss_val:.4f}")
    return final_image, final_loss_val