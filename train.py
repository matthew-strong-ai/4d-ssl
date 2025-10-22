import argparse
import os
import torch
import torch.nn.functional as F

import numpy as np

from accelerate import Accelerator
from tqdm import tqdm
import torchvision.transforms as T
import gc
from torch.utils.tensorboard import SummaryWriter


from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import AutonomySSLModel
from SpaTrackerV2.models.SpaTrackV2.models.predictor import Predictor
from SpaTrackerV2.models.SpaTrackV2.models.utils import get_points_on_a_grid
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from SpaTrackerV2.models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import depth_to_points_colmap

# Import depth loss functions
from SpaTrackerV2.models.moge.train.losses import (
    affine_invariant_global_loss,
    edge_loss,
    normal_loss,
    mask_l2_loss,
)

from SpaTrackerV2.multi_folder_consecutive_images_dataset import MultiFolderConsecutiveImagesDataset, get_default_transforms


def compute_loss(predictions, targets):
    """
    Compute the loss between autonomy ssl model and pseudo labels from full spatial tracker.
    Includes both pose loss and depth loss computation.
    """
    gt_extrinsics = targets["extrinsics"]
    # add batch dim
    gt_extrinsics = gt_extrinsics.unsqueeze(0) if gt_extrinsics.ndim == 3 else gt_extrinsics
    gt_intrinsics = targets["intrinsics"]
    # add batch dim
    gt_intrinsics = gt_intrinsics.unsqueeze(0) if gt_intrinsics.ndim == 3 else gt_intrinsics

    # send to device
    device = predictions["images"].device
    gt_extrinsics = gt_extrinsics.to(device)
    gt_intrinsics = gt_intrinsics.to(device)

    # Get image dimensions
    B, T, C, H, W = predictions["images"].shape
    H_resize, W_resize = H, W

    # =================== DEPTH LOSS COMPUTATION ===================
    depth_loss = 0

    if "gt_depth" in targets and "points_map" in predictions:
        gt_depth = targets["gt_depth"]
        pred_points_map = predictions["points_map"]
        
        # Ensure gt_depth has correct dimensions and is on device
        if gt_depth.ndim == 3:  # (T, H, W)
            # gt_depth = gt_depth.unsqueeze(0)  # (1, T, H, W)
            gt_depth = gt_depth.view(B_gt * T_gt, 1, H_gt, W_gt)  # (B*T, 1, H, W)

        # Reshape to (B*T, 1, H, W) for processing
        if gt_depth.ndim == 4:  # (B, T, H, W)
            B_gt, T_gt, H_gt, W_gt = gt_depth.shape
            gt_depth = gt_depth.view(B_gt * T_gt, 1, H_gt, W_gt)  # (B*T, 1, H, W)
        
        gt_depth = gt_depth.to(device)

        import pdb; pdb.set_trace()  # Debugging breakpoint --- IGNORE ---
        
        # Resize ground truth depth if needed
        if gt_depth.shape[-2:] != (H_resize, W_resize):
            gt_depth = F.interpolate(gt_depth, size=(H_resize, W_resize), mode='nearest')
        
        # Create depth masks (similar to loss.py)
        valid_depth_mask = gt_depth > 0
        if valid_depth_mask.sum() > 0:
            _depths = gt_depth[valid_depth_mask].reshape(-1)
            if len(_depths) > 4:  # Ensure we have enough points for quantile computation
                q25_idx = int(0.25 * len(_depths))
                q75_idx = int(0.75 * len(_depths))
                q25 = torch.kthvalue(_depths, max(1, q25_idx)).values
                q75 = torch.kthvalue(_depths, max(1, q75_idx)).values
                iqr = q75 - q25
                upper_bound = (q75 + 0.8*iqr).clamp(min=1e-6, max=10*q25)
                _depth_roi = torch.tensor([1e-1, upper_bound.item()], dtype=gt_depth.dtype, device=gt_depth.device)
                
                # Create masks
                mask_roi = (gt_depth > _depth_roi[0]) & (gt_depth < _depth_roi[1])
                gt_mask_fin = ((gt_depth > 0) * mask_roi).float()
                
                # Filter the sky
                inf_thres = 50*q25.clamp(min=200, max=1e3)
                gt_mask_inf = (gt_depth > inf_thres).float()
                
                # Final ground truth mask
                gt_mask = (gt_depth > 0) * (gt_depth < 10*q25)
                gt_mask = gt_mask.float()
                
                # Convert ground truth depth to point cloud
                intrinsics_reshaped = gt_intrinsics.view(B*T, 3, 3)
                points_map_gt = depth_to_points_colmap(gt_depth.squeeze(1), intrinsics_reshaped)
                
                # Ensure predicted and GT point maps have same spatial dimensions
                pred_points_shape = pred_points_map.shape  # Expected: (B*T, H, W, 3)
                gt_points_shape = points_map_gt.shape      # Should match: (B*T, H, W, 3)
                
                if pred_points_shape[-3:-1] != gt_points_shape[-3:-1]:
                    # Resize GT to match prediction resolution
                    H_pred, W_pred = pred_points_shape[-3:-1]
                    H_gt, W_gt = gt_points_shape[-3:-1]
                    print(f"Resizing GT points from {(H_gt, W_gt)} to {(H_pred, W_pred)}")
                    
                    # Reshape for interpolation: (B*T, H, W, 3) -> (B*T, 3, H, W)
                    points_map_gt_reshaped = points_map_gt.permute(0, 3, 1, 2)
                    gt_mask_reshaped = gt_mask.permute(0, 3, 1, 2) if gt_mask.shape[-1] == 1 else gt_mask.unsqueeze(1)
                    
                    # Resize using interpolation
                    points_map_gt = F.interpolate(points_map_gt_reshaped, size=(H_pred, W_pred), mode='bilinear', align_corners=False)
                    gt_mask = F.interpolate(gt_mask_reshaped.float(), size=(H_pred, W_pred), mode='nearest')
                    
                    # Reshape back: (B*T, 3, H, W) -> (B*T, H, W, 3)
                    points_map_gt = points_map_gt.permute(0, 2, 3, 1)
                    gt_mask = gt_mask.permute(0, 2, 3, 1) if gt_mask.shape[1] == 1 else gt_mask.squeeze(1)
                
                # Compute depth losses (similar to loss.py but scaled for training)
                try:
                    # Global invariant loss - handle different return values safely

                    import pdb; pdb.set_trace()
                    loss_result = affine_invariant_global_loss(
                        pred_points_map, points_map_gt, gt_mask[:,0], align_resolution=32
                    )
                    # Unpack safely - function may return different number of values
                    ln_depth_glob = loss_result[0]
                    ln_depth_glob = ln_depth_glob.mean() * 10.0  # Scaled down from 100 in loss.py
                    
                    if len(loss_result) >= 3:
                        gt_metric_scale = loss_result[2]
                    else:
                        gt_metric_scale = torch.tensor(1.0, device=device)
                    
                    if len(loss_result) >= 4:
                        gt_metric_shift = loss_result[3]
                    else:
                        gt_metric_shift = torch.zeros_like(gt_metric_scale)
                    
                    
                    # Edge loss
                    ln_edge, _ = edge_loss(pred_points_map, points_map_gt, gt_mask[:,0])
                    ln_edge = ln_edge.mean() * 0.1  # Scaled down
                    
                    # Normal loss 
                    ln_normal, _ = normal_loss(pred_points_map, points_map_gt, gt_mask[:,0])
                    ln_normal = ln_normal.mean() * 0.1  # Scaled down
                    
                    # Consistency loss (simplified version)
                    norm_rescale = gt_metric_scale.mean()
                    points_map_gt_cons = points_map_gt.clone() / norm_rescale
                    pred_mask = predictions.get("unc_metric", torch.ones_like(pred_points_map[..., 0])).clamp(min=5e-2)
                    ln_cons = torch.abs(pred_points_map - points_map_gt_cons).norm(dim=-1) * pred_mask
                    ln_cons = ln_cons[(1-gt_mask_inf.squeeze()).bool()].clamp(max=100).mean() * 0.05  # Scaled down
                    
                    depth_loss = ln_depth_glob + ln_edge + ln_normal + ln_cons
                    
                except Exception as e:
                    print(f"Warning: Depth loss computation failed: {e}")
                    depth_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                print("Warning: Not enough valid depth points for loss computation")
                depth_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            print("Warning: No valid depth points found")
            depth_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # =================== POSE LOSS COMPUTATION ===================
    # compute camera loss; the adjustment in camera motion should not be too significant 
    # (may need to make encoder more expressive and have smarter attention mechanism)
    ln_pose = 0
    for i_t, pose_enc_i in enumerate(predictions["pose_enc_list"]):
        pose_enc_gt = extri_intri_to_pose_encoding(torch.inverse(gt_extrinsics)[...,:3,:4], gt_intrinsics, predictions["images"].shape[-2:])
        T_loss = torch.abs(pose_enc_i[..., :3] - pose_enc_gt[..., :3]).mean()
        R_loss = torch.abs(pose_enc_i[..., 3:7] - pose_enc_gt[..., 3:7]).mean()
        K_loss = torch.abs(pose_enc_i[..., 7:] - pose_enc_gt[..., 7:]).mean()
        pose_loss_i = 25*(T_loss + R_loss) + K_loss
        ln_pose += 0.8**(len(predictions["pose_enc_list"]) - i_t - 1)*(pose_loss_i)
    ln_pose = 0.001*ln_pose

    # =================== TOTAL LOSS ===================
    total_loss = ln_pose + depth_loss
    
    return total_loss

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Training script with Accelerate")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing subfolders of images")
    parser.add_argument("--batch_size", type=int, default=20,
                      help="Batch size for training (number of consecutive images per batch)")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--val_freq", type=int, default=1000,
                      help="Validate every N steps")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for query points")
    parser.add_argument("--n_visualize", type=int, default=3, help="Number of random batches to visualize before training")

    args = parser.parse_args()

    accelerator = Accelerator()
    grid_size = args.grid_size

    # Find all subdirectories in root_dir
    image_dirs = [os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                  if os.path.isdir(os.path.join(args.root_dir, d))]
    print(f"Found {len(image_dirs)} subfolders:")
    for d in image_dirs:
        print(f"  {d}")


    # Load dataset
    dataset = MultiFolderConsecutiveImagesDataset(
        image_dirs=image_dirs,
        batch_size=args.batch_size,
        transform=get_default_transforms()
    )

    # Optionally visualize a few random batches
    # from SpaTrackerV2.multi_folder_consecutive_images_dataset import visualize_random_samples
    # visualize_random_samples(dataset, n=args.n_visualize)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Each item is a batch of images (T, C, H, W)
        shuffle=True,
        num_workers=2
    )

    track_mode = "offline"  # or "online", depending on your use case

    if track_mode == "offline":
        tracking_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        tracking_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")


    tracking_model.eval()
    tracking_model.to("cuda")
    # Initialize AutonomySSLModel from pretrained spatial tracker
    model = AutonomySSLModel.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare objects with accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader
    )

    device = accelerator.device

    # TensorBoard SummaryWriter
    writer = SummaryWriter()

    # Training loop
    total_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch}", 
            disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(progress_bar):

            # batch: (1, T, C, H, W)
            # remove first dim
            batch = batch.squeeze(0)  # (T, C, H, W)
            video_tensor = preprocess_image(batch)[None]  # (1, T, C, H, W)

            prediction_dict = {}
            gt_dict = {}

            # get prediction from model, before point track refinement
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = model(video_tensor.to(device))
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
                # Save all points_map, not just depth
                points_map = predictions["points_map"]


            # get copy and detach
            depth_tensor = depth_map.detach().squeeze().cpu().numpy()
            extrs = extrinsic.detach().squeeze().cpu().numpy()

            depth_tensor = depth_map.detach().squeeze().cpu().numpy()
            extrs = extrinsic.detach().squeeze().cpu().numpy()
            # extr_file is inverse of extrs, using numpy of extrs
            extrs_inv = np.linalg.inv(extrs)
            intrs = intrinsic.detach().squeeze().cpu().numpy()
            # video_tensor = video_tensor.squeeze()
            unc_metric = depth_conf.detach().squeeze().cpu().numpy() > 0.5
            conf = depth_conf.detach().squeeze().cpu().numpy()
            depth_tensor_write = depth_tensor.copy()
            depth_tensor_write[conf<0.5] = 0

            # data_npz_load = {}
            # # load up data npz load
            # data_npz_load["extrinsics"] = extrs_inv
            # data_npz_load["intrinsics"] = intrs
            # # depth_save = points_map[:,2,...]
            # # depth_save[conf<0.5] = 0
            # data_npz_load["depths"] = depth_tensor_write
            # data_npz_load["video"] = (video_tensor.squeeze()).cpu().numpy()
            # data_npz_load["unc_metric"] = conf
            # # save the data_npz_load to a npz file
            # output_file = f"output_{total_step}.npz"
            # total_step += 1
            # np.savez(output_file, **data_npz_load)
            # print(f"Saved data to {output_file}")

            # load up prediction dict
            prediction_dict["extrinsics"] = extrinsic
            prediction_dict["intrinsics"] = intrinsic
            prediction_dict["pose_enc_list"] = predictions.get("pose_enc_list", [])
            prediction_dict["images"] = predictions.get("images", None)
            prediction_dict["points_map"] = points_map  
            prediction_dict["unc_metric"] = depth_conf

            frame_H, frame_W = video_tensor.shape[3:]
            grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
            query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

            video_tensor_track = video_tensor.squeeze().to(device)

            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    (
                        c2w_traj, intrs, point_map, conf_depth,
                        track3d_pred, track2d_pred, vis_pred, conf_pred, video
                    ) = tracking_model.forward(video_tensor_track, depth=depth_tensor,
                                        intrs=intrs, extrs=extrs,
                                        queries=query_xyt,
                                        fps=1, full_point=False, iters_track=4,
                                        query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                                        support_frame=len(video_tensor_track)-1, replace_ratio=0.2)
                tracking_error = False
            except Exception as e:
                print(f"Tracking model error at step {step}: {e}")
                tracking_error = True

            if tracking_error:
                print("Skipping loss and backward for this batch due to tracking error.")
                continue

            gt_depth = point_map[:,2,...].clone()


            gt_dict["gt_depth"] = gt_depth
            gt_dict["gt_depth_conf"] = conf_depth

            max_size = 336
            h, w = video.shape[2:]
            video_tensor = video_tensor_track

            scale = min(max_size / h, max_size / w)
            if scale < 1:
                new_h, new_w = int(h * scale), int(w * scale)
                video = T.Resize((new_h, new_w))(video)
                video_tensor = T.Resize((new_h, new_w))(video_tensor)
                point_map = T.Resize((new_h, new_w))(point_map)
                conf_depth = T.Resize((new_h, new_w))(conf_depth)
                track2d_pred[...,:2] = track2d_pred[...,:2] * scale
                intrs[:,:2,:] = intrs[:,:2,:] * scale
                if depth_tensor is not None:
                    if isinstance(depth_tensor, torch.Tensor):
                        depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                    else:
                        depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))


            gt_dict["extrinsics"] = c2w_traj
            gt_dict["intrinsics"] = intrs

            # Use a dummy loss for demonstration (replace with your own)
            data_npz_load = {}
            data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
            data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
            data_npz_load["intrinsics"] = intrs.cpu().numpy()
            depth_save = point_map[:,2,...]
            depth_save[conf_depth<0.5] = 0
            data_npz_load["depths"] = depth_save.cpu().numpy()
            data_npz_load["video"] = (video_tensor).cpu().numpy()
            data_npz_load["visibs"] = vis_pred.cpu().numpy()
            data_npz_load["unc_metric"] = conf_depth.cpu().numpy()

            # save the data_npz_load to a npz file
            output_file = f"output_{total_step}.npz"
            np.savez(output_file, **data_npz_load)
            print(f"Saved data to {output_file}")

            loss = compute_loss(prediction_dict, gt_dict)

            # # Log loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), total_step)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                progress_bar.set_postfix({"loss": loss.item()})
            total_step += 1

            # clear gpu
            torch.cuda.empty_cache()
            gc.collect()
            
            # Visualize the depth map (depth_save)
            # import matplotlib.pyplot as plt
            # dm = depth_save.cpu().numpy() if hasattr(depth_save, 'cpu') else np.array(depth_save)
            # if dm.ndim == 3:  # (T, H, W)
            #     nframes = dm.shape[0]
            #     ncols = min(5, nframes)
            #     nrows = (nframes + ncols - 1) // ncols
            #     vmin = np.percentile(dm, 2)
            #     vmax = np.percentile(dm, 98)
            #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
            #     axes = np.array(axes).reshape(nrows, ncols)
            #     for i in range(nframes):
            #         row, col = divmod(i, ncols)
            #         ax = axes[row, col]
            #         ax.imshow(dm[i], cmap='plasma', vmin=vmin, vmax=vmax)
            #         ax.axis('off')
            #         ax.set_title(f'Depth Frame {i}')
            #     for i in range(nframes, nrows * ncols):
            #         row, col = divmod(i, ncols)
            #         axes[row, col].axis('off')
            #     plt.suptitle("Depth map visualization (depth_save)")
            #     plt.tight_layout()
            #     plt.show()
            #     input("Above: depth map visualization. Press Enter to continue...")

        # Save model checkpoint at the end of each epoch
        accelerator.save_state(f"checkpoint_epoch_{epoch}.pt")
        print(f"Model checkpoint saved: checkpoint_epoch_{epoch}.pt")

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
