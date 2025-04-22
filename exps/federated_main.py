#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
import os # <-- Add os import
import matplotlib.pyplot as plt # <-- Add plt import

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist, CNNCifar # <-- Make sure all needed models are imported
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
# --- Import the new attack function ---
from attack import prototype_inversion_attack #<-- ADD: Assuming attack code is in attack.py in the same dir

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# --- ADD: Function to save attack artifacts ---
def save_attack_artifacts(round_num, models, protos, base_path='attack_artifacts'):
    """Saves models and prototypes for a specific round."""
    round_path = os.path.join(base_path, f'round_{round_num}')
    os.makedirs(round_path, exist_ok=True)

    # Save models
    for idx, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(round_path, f'client_{idx}_model.pth'))

    # Save local prototypes (dictionary)
    # Note: Prototypes are tensors, handle potential CUDA tensors
    protos_to_save = {}
    if protos: # Check if protos dictionary is not empty
        for client_idx, proto_dict in protos.items():
             # Ensure client_idx is within the range of models being saved
            if client_idx < len(models):
                protos_to_save[client_idx] = {
                    label: p.clone().detach().cpu() for label, p in proto_dict.items()
                }
            else:
                print(f"Warning: Client index {client_idx} in protos is out of range for saved models (num_models={len(models)}). Skipping.")

    np.save(os.path.join(round_path, 'local_protos.npy'), protos_to_save)
    print(f"Saved attack artifacts for round {round_num} to {round_path}")


# --- Modify FedProto_taskheter ---
def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)
    active_model_list = copy.deepcopy(local_model_list) # Keep track of models that successfully load state dicts

    train_loss, train_accuracy = [], []

    # --- ADD: Configuration for saving artifacts ---
    save_interval = 25 # Save every 25 rounds (adjust as needed)
    attack_artifact_path = f'./attack_artifacts_{args.dataset}_{args.mode}'
    os.makedirs(attack_artifact_path, exist_ok=True)
    # --- End ADD ---

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos_agg = [], [], {} # Renamed local_protos to local_protos_agg
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0

        current_round_participants = idxs_users # Assuming all users participate each round
        successfully_updated_models = [] # List to hold models updated this round

        for idx in current_round_participants:
            # Use the latest successfully updated model state for this client
            model_to_train = copy.deepcopy(active_model_list[idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])

            w, loss, acc, agg_protos_label = local_model.update_weights_het(
                args, idx, global_protos, model=model_to_train, global_round=round
            )

            # Calculate final local protos for client idx for this round
            agg_protos = agg_func(agg_protos_label)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos_agg[idx] = agg_protos # Store the aggregated proto for the client

            # Keep track of the model state after local training
            temp_model = copy.deepcopy(active_model_list[idx]) # Get architecture
            temp_model.load_state_dict(w) # Load updated weights
            successfully_updated_models.append(temp_model)

            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']


        # --- Update the active models list ---
        # This replaces the potentially problematic direct load_state_dict loop
        if len(successfully_updated_models) == len(current_round_participants):
             active_model_list = copy.deepcopy(successfully_updated_models)
        else:
             # Handle cases where some clients might fail (though not explicitly modeled here yet)
             print("Warning: Mismatch in number of updated models. Check client participation.")
             # Fallback or specific error handling needed if participation varies


        # --- ADD: Save artifacts periodically ---
        # Save the state of models *after* local training for this round
        if (round + 1) % save_interval == 0 or round == args.rounds - 1:
             # Pass the models list corresponding to this round's state and the computed local protos
            save_attack_artifacts(round + 1, active_model_list, local_protos_agg, base_path=attack_artifact_path)
        # --- End ADD ---

        # update global protos using the aggregated local ones
        global_protos = proto_aggregation(local_protos_agg)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # --- Final Testing ---
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, active_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    if loss_list: # Check if loss_list is not empty
        print('For all users (with protos), mean of proto loss is {:.5f}, std of proto loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos for visualization if needed (original functionality)
    if args.dataset == 'mnist':
        save_protos(args, active_model_list, test_dataset, user_groups_lt)


    # --- ADD: Optional: Run attack after training is complete ---
    print("\n--- Running Post-Training Prototype Inversion Attack ---")
    # Determine the round to attack (last saved round)
    last_saved_round = args.rounds - ((args.rounds-1) % save_interval) if args.rounds >0 else 0
    if last_saved_round <= 0: last_saved_round = min(save_interval, args.rounds) # Handle edge cases
    if last_saved_round > args.rounds: last_saved_round = args.rounds # Clamp to max rounds

    target_round = last_saved_round
    target_client_idx = 0 # Attack client 0
    target_class_label = -1 # Placeholder

    print(f"Attempting attack on artifacts from round {target_round}...")
    artifact_round_path = os.path.join(attack_artifact_path, f'round_{target_round}')

    if not os.path.exists(artifact_round_path):
         print(f"Artifact directory not found: {artifact_round_path}. Skipping post-training attack.")
    else:
        try:
            # --- Load Model ---
            model_path = os.path.join(artifact_round_path, f'client_{target_client_idx}_model.pth')
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found: {model_path}")

            # Re-build the model architecture based on args for the target client
            # This needs to match the architecture saved.
            print(f"Loading model architecture for client {target_client_idx}...")
            # --- This block needs to exactly replicate the model creation in the main block ---
            temp_args = copy.deepcopy(args) # Use a copy of args to avoid side effects
            if temp_args.dataset == 'mnist':
                if temp_args.mode == 'model_heter':
                    if target_client_idx<7: temp_args.out_channels = 18
                    elif target_client_idx>=7 and target_client_idx<14: temp_args.out_channels = 20
                    else: temp_args.out_channels = 22
                else: temp_args.out_channels = 20
                target_model_instance = CNNMnist(args=temp_args)
            elif temp_args.dataset == 'femnist':
                 if temp_args.mode == 'model_heter':
                    if target_client_idx<7: temp_args.out_channels = 18
                    elif target_client_idx>=7 and target_client_idx<14: temp_args.out_channels = 20
                    else: temp_args.out_channels = 22
                 else: temp_args.out_channels = 20
                 target_model_instance = CNNFemnist(args=temp_args)
            elif temp_args.dataset == 'cifar100' or temp_args.dataset == 'cifar10':
                if temp_args.mode == 'model_heter':
                    if target_client_idx<10: temp_args.stride = [1,4]
                    else: temp_args.stride = [2,2]
                else: temp_args.stride = [2, 2]
                # Ensure num_classes matches the model definition if needed
                target_model_instance = resnet18(args=temp_args, pretrained=False, num_classes=temp_args.num_classes)
                # Note: ResNet loading in main() uses pretrained weights partially, replicate if necessary
                # Here, we just load the saved state dict fully.
            else:
                raise ValueError(f"Unsupported dataset for attack model loading: {temp_args.dataset}")
            # --- End model creation replication ---

            target_model_instance.load_state_dict(torch.load(model_path, map_location=args.device))
            target_model_instance.to(args.device)
            target_model_instance.eval()
            print(f"Successfully loaded model for client {target_client_idx} from round {target_round}.")


            # --- Load Prototypes ---
            proto_path = os.path.join(artifact_round_path, 'local_protos.npy')
            if not os.path.exists(proto_path):
                raise FileNotFoundError(f"Prototype file not found: {proto_path}")

            saved_local_protos = np.load(proto_path, allow_pickle=True).item()

            if target_client_idx in saved_local_protos and saved_local_protos[target_client_idx]:
                available_classes = list(saved_local_protos[target_client_idx].keys())
                if available_classes:
                    target_class_label = random.choice(available_classes) # Attack a random available class
                    print(f"Attacking Round {target_round}, Client {target_client_idx}, Class {target_class_label}")

                    target_prototype = saved_local_protos[target_client_idx][target_class_label]
                    # Ensure it's a tensor on the correct device
                    target_prototype = torch.tensor(target_prototype, dtype=torch.float32).to(args.device)

                    # --- Determine Input Shape ---
                    if args.dataset == 'mnist' or args.dataset == 'femnist':
                        input_shape = (args.num_channels, 28, 28)
                    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
                        input_shape = (3, 32, 32) # CIFAR uses 3 channels
                    else:
                        raise ValueError("Unknown dataset for input shape determination")

                    # --- Run the Attack ---
                    reconstructed_image, final_loss = prototype_inversion_attack(
                        target_model=target_model_instance,
                        target_prototype=target_prototype,
                        input_shape=input_shape,
                        args=args,
                        num_iterations=500, # Adjust iterations
                        lr=0.05, # Adjust learning rate (0.01 or 0.1 might work)
                        use_tv_loss=True # Use Total Variation Regularization
                    )

                    # --- Visualize Result ---
                    plt.figure(figsize=(4,4))
                    # Adjust imshow based on channels
                    if reconstructed_image.shape[1] == 1: # Grayscale
                        plt.imshow(reconstructed_image.squeeze().cpu().numpy(), cmap='gray')
                    else: # RGB
                         # Need to transpose channels from (C, H, W) to (H, W, C) for imshow
                        plt.imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())
                    plt.title(f'Recon: R{target_round},C{target_client_idx},Cls{target_class_label}\nLoss:{final_loss:.4f}', fontsize=10)
                    plt.axis('off')
                    save_path = os.path.join(artifact_round_path, f'reconstruction_c{target_client_idx}_cls{target_class_label}.png')
                    plt.savefig(save_path)
                    plt.show() # Optionally show plot interactively
                    plt.close() # Close the figure
                    print(f"Reconstruction visualization saved to {save_path}")

                else:
                    print(f"Client {target_client_idx} had no prototypes saved in round {target_round}.")
            else:
                 print(f"No prototypes found for client {target_client_idx} in round {target_round}.")

        except FileNotFoundError as e:
            print(f"Error loading artifacts: {e}. Skipping post-training attack.")
        except Exception as e:
            print(f"An error occurred during the post-training attack: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
    # --- End ADD ---


# --- ADD definition for FedProto_modelheter ---
# Make sure this function also includes the artifact saving and post-training attack logic
# similar to FedProto_taskheter
def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    # This function needs the same modifications as FedProto_taskheter:
    # 1. Add save_interval, attack_artifact_path setup
    # 2. Modify the training loop to use active_model_list
    # 3. Call save_attack_artifacts periodically inside the loop
    # 4. Add the post-training attack execution block at the end

    # ----- Start Placeholder for FedProto_modelheter -----
    print("FedProto_modelheter function needs modifications similar to FedProto_taskheter for attack implementation.")
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)
    active_model_list = copy.deepcopy(local_model_list)
    train_loss, train_accuracy = [], []

    # --- ADD: Configuration for saving artifacts ---
    save_interval = 25 # Save every 25 rounds (adjust as needed)
    attack_artifact_path = f'./attack_artifacts_{args.dataset}_{args.mode}' # Use _mh suffix
    os.makedirs(attack_artifact_path, exist_ok=True)
    # --- End ADD ---

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos_agg = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')
        proto_loss = 0
        current_round_participants = idxs_users
        successfully_updated_models = []

        for idx in current_round_participants:
            model_to_train = copy.deepcopy(active_model_list[idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, agg_protos_label = local_model.update_weights_het(
                args, idx, global_protos, model=model_to_train, global_round=round
            )
            agg_protos = agg_func(agg_protos_label)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos_agg[idx] = agg_protos
            temp_model = copy.deepcopy(active_model_list[idx])
            temp_model.load_state_dict(w)
            successfully_updated_models.append(temp_model)
            # Add summary writer lines if needed...
            proto_loss += loss['2']

        # --- Update the active models list ---
        if len(successfully_updated_models) == len(current_round_participants):
             active_model_list = copy.deepcopy(successfully_updated_models)
        else:
             print("Warning: Mismatch in number of updated models (model_heter).")


        # --- ADD: Save artifacts periodically ---
        if (round + 1) % save_interval == 0 or round == args.rounds - 1:
            save_attack_artifacts(round + 1, active_model_list, local_protos_agg, base_path=attack_artifact_path)
        # --- End ADD ---

        global_protos = proto_aggregation(local_protos_agg)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # --- Final Testing ---
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, active_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('[Model Heter] For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('[Model Heter] For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    if loss_list:
         print('[Model Heter] For all users (with protos), mean of proto loss is {:.5f}, std of proto loss is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))


    # --- ADD: Optional: Run attack after training is complete (Copy from FedProto_taskheter) ---
    print("\n--- Running Post-Training Prototype Inversion Attack (Model Heter) ---")
    last_saved_round = args.rounds - ((args.rounds-1) % save_interval) if args.rounds >0 else 0
    if last_saved_round <= 0: last_saved_round = min(save_interval, args.rounds)
    if last_saved_round > args.rounds: last_saved_round = args.rounds
    target_round = last_saved_round
    target_client_idx = 0
    target_class_label = -1
    print(f"Attempting attack on artifacts from round {target_round}...")
    artifact_round_path = os.path.join(attack_artifact_path, f'round_{target_round}')
    # ... (Copy the entire try-except block for loading artifacts and running the attack from FedProto_taskheter) ...
    # --- Make sure to use the correct 'attack_artifact_path' and potentially adjust model loading if needed ---
    if not os.path.exists(artifact_round_path):
         print(f"Artifact directory not found: {artifact_round_path}. Skipping post-training attack.")
    else:
        try:
            # --- Load Model ---
            model_path = os.path.join(artifact_round_path, f'client_{target_client_idx}_model.pth')
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"Loading model architecture for client {target_client_idx} (mode: {args.mode})...")
            temp_args = copy.deepcopy(args)
            # --- Replication of model creation for model_heter ---
            if temp_args.dataset == 'mnist':
                if temp_args.mode == 'model_heter':
                    if target_client_idx<7: temp_args.out_channels = 18
                    elif target_client_idx>=7 and target_client_idx<14: temp_args.out_channels = 20
                    else: temp_args.out_channels = 22
                else: temp_args.out_channels = 20 # Fallback
                target_model_instance = CNNMnist(args=temp_args)
            elif temp_args.dataset == 'femnist':
                 if temp_args.mode == 'model_heter':
                    if target_client_idx<7: temp_args.out_channels = 18
                    elif target_client_idx>=7 and target_client_idx<14: temp_args.out_channels = 20
                    else: temp_args.out_channels = 22
                 else: temp_args.out_channels = 20 # Fallback
                 target_model_instance = CNNFemnist(args=temp_args)
            elif temp_args.dataset == 'cifar100' or temp_args.dataset == 'cifar10':
                if temp_args.mode == 'model_heter':
                    if target_client_idx<10: temp_args.stride = [1,4]
                    else: temp_args.stride = [2,2]
                else: temp_args.stride = [2, 2] # Fallback
                target_model_instance = resnet18(args=temp_args, pretrained=False, num_classes=temp_args.num_classes)
            else:
                raise ValueError(f"Unsupported dataset for attack model loading: {temp_args.dataset}")
            # --- End model creation replication ---
            target_model_instance.load_state_dict(torch.load(model_path, map_location=args.device))
            target_model_instance.to(args.device)
            target_model_instance.eval()
            print(f"Successfully loaded model for client {target_client_idx} from round {target_round}.")

            # --- Load Prototypes and Run Attack (identical logic to task_heter version) ---
            proto_path = os.path.join(artifact_round_path, 'local_protos.npy')
            if not os.path.exists(proto_path): raise FileNotFoundError(f"Prototype file not found: {proto_path}")
            saved_local_protos = np.load(proto_path, allow_pickle=True).item()
            if target_client_idx in saved_local_protos and saved_local_protos[target_client_idx]:
                available_classes = list(saved_local_protos[target_client_idx].keys())
                if available_classes:
                    target_class_label = random.choice(available_classes)
                    print(f"Attacking Round {target_round}, Client {target_client_idx}, Class {target_class_label}")
                    target_prototype = saved_local_protos[target_client_idx][target_class_label]
                    target_prototype = torch.tensor(target_prototype, dtype=torch.float32).to(args.device)
                    # Determine Input Shape
                    if args.dataset == 'mnist' or args.dataset == 'femnist': input_shape = (args.num_channels, 28, 28)
                    elif args.dataset == 'cifar10' or args.dataset == 'cifar100': input_shape = (3, 32, 32)
                    else: raise ValueError("Unknown dataset for input shape")
                    # Run Attack
                    reconstructed_image, final_loss = prototype_inversion_attack(
                        target_model=target_model_instance, target_prototype=target_prototype, input_shape=input_shape, args=args,
                        num_iterations=500, lr=0.05, use_tv_loss=True)
                    # Visualize
                    plt.figure(figsize=(4,4));
                    if reconstructed_image.shape[1] == 1: plt.imshow(reconstructed_image.squeeze().cpu().numpy(), cmap='gray')
                    else: plt.imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())
                    plt.title(f'ReconMH: R{target_round},C{target_client_idx},Cls{target_class_label}\nLoss:{final_loss:.4f}', fontsize=10)
                    plt.axis('off'); save_path = os.path.join(artifact_round_path, f'reconstruction_mh_c{target_client_idx}_cls{target_class_label}.png');
                    plt.savefig(save_path); plt.show(); plt.close(); print(f"Reconstruction visualization saved to {save_path}")
                else: print(f"Client {target_client_idx} had no protos in round {target_round}.")
            else: print(f"No protos found for client {target_client_idx} in round {target_round}.")
        except FileNotFoundError as e: print(f"Error loading artifacts: {e}. Skipping attack.")
        except Exception as e: print(f"Error during attack: {e}"); import traceback; traceback.print_exc()
    # --- End ADD ---
    # ----- End Placeholder for FedProto_modelheter -----


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        print(f"Using GPU: {args.gpu}")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        print("Using CPU")
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(max(1, args.shots - args.stdev), args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(max(1, args.shots - args.stdev), args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users) # Original code narrow range
        k_list = np.clip(k_list, 1, None) # Ensure k is at least 1
    elif args.dataset == 'femnist':
        k_list = np.random.randint(max(1, args.shots - args.stdev), args.shots + args.stdev + 1, args.num_users)
    else:
        k_list = np.random.randint(max(1, args.shots - args.stdev), args.shots + args.stdev + 1, args.num_users) # Default if dataset unknown

    print("Client ways (n_list):", n_list)
    print("Client shots (k_list):", k_list)
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # Build models list
    local_model_list = []
    print("Building models for {} users...".format(args.num_users))
    for i in range(args.num_users):
        temp_args = copy.deepcopy(args) # Use a copy of args for each client
        temp_args.num_classes = len(classes_list[i]) # Set num_classes based on the actual classes for the client

        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7: temp_args.out_channels = 18
                elif i>=7 and i<14: temp_args.out_channels = 20
                else: temp_args.out_channels = 22
            else: temp_args.out_channels = 20
            local_model = CNNMnist(args=temp_args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7: temp_args.out_channels = 18
                elif i>=7 and i<14: temp_args.out_channels = 20
                else: temp_args.out_channels = 22
            else: temp_args.out_channels = 20
            local_model = CNNFemnist(args=temp_args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i<10: temp_args.stride = [1,4]
                else: temp_args.stride = [2,2]
            else: temp_args.stride = [2, 2]

            # Important: num_classes for ResNet should match the specific client's classes count
            resnet = resnet18(args=temp_args, pretrained=False, num_classes=temp_args.num_classes)
            try:
                initial_weight = model_zoo.load_url(model_urls['resnet18'], progress=False)
                local_model = resnet
                initial_weight_1 = local_model.state_dict()
                # Load pretrained weights except for the final layer (fc) and potentially conv1/bn1
                for key in initial_weight.keys():
                    if key in initial_weight_1 and not key.startswith('fc.'): # Load if key matches and not FC layer
                         # Optionally skip conv1/bn1 if input channels mismatch, though ResNet usually starts with 3 channels
                         # if not (key.startswith('conv1') or key.startswith('bn1')):
                         initial_weight_1[key] = initial_weight[key]

                local_model.load_state_dict(initial_weight_1)
                print(f"Loaded partial pretrained weights for ResNet18 client {i}")
            except Exception as e:
                 print(f"Could not load pretrained weights for ResNet client {i}: {e}. Using randomly initialized model.")
                 local_model = resnet # Use the initialized ResNet

        else:
             raise NotImplementedError(f"Dataset {args.dataset} model build not implemented.")

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    # Run training and attack based on mode
    if args.mode == 'task_heter':
        print("\n--- Starting FedProto Task Heterogeneity Training ---")
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    elif args.mode == 'model_heter':
        print("\n--- Starting FedProto Model Heterogeneity Training ---")
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
         print(f"Unknown mode: {args.mode}. Exiting.")


    print(f'\nTotal Run Time: {time.time()-start_time:.2f}s')