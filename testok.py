#从modem那里偷的test，可以用，测试的没有那么准确，但是已经是目前最好的结果了。
#可以测试图片，也可以测试psnr和ssim。位置正确不要乱改了哈哈
#在settingI目录下进行测试，命令如下，自行修改！
#python testok.py --ckpt_path ./experiments/finetune_HOGformer_Allweather_after_DCPT/model/net_g_best_psnr_33.0230_ssim_0.9466.pth --input_folder ./Allweather/test/Snow100K-Test/Snow100K-S/synthetic --gt_folder ./Allweather/test/Snow100K-Test/Snow100K-S/gt --output_folder ./results/S-S
#python testok.py --ckpt_path ./experiments/finetune_HOGformer_Allweather_after_DCPT/model/net_g_best_psnr_33.0230_ssim_0.9466.pth --input_folder ./Allweather/test/RainDrop/test_a/input --gt_folder ./Allweather/test/RainDrop/test_a/gt --output_folder ./results/RainDrop
#python testok.py --ckpt_path ./experiments/finetune_HOGformer_Allweather_after_DCPT/model/net_g_best_psnr_33.0230_ssim_0.9466.pth --input_folder ./Allweather/test/Test1/input --gt_folder ./Allweather/test/Test1/gt --output_folder ./results/RainDrop
#SPM+CPR
import argparse
import os
import random
import time
import cv2
import glob
import torch
import numpy as np
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from natsort import natsorted
from basicsr.models.archs.hogformer_arch import HOGformer
import shutil


def main():
    print('Loading model...')
    model = HOGformer(
        inp_channels=3,
        out_channels=3,
        dim=36,
        num_blocks=[4, 4, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.667,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ).cuda()
    
    print(f'Loading checkpoint from {opt.ckpt_path}')
    checkpoint = torch.load(opt.ckpt_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    
    model.eval()
    
    psnr_list = []
    ssim_list = []
    processing_times = []
    
    input_files = natsorted(glob.glob(os.path.join(opt.input_folder, '*')))
    total_files = len(input_files)
    
    print(f'Found {total_files} images to process')
    print('='*60)
    
    for idx, path in enumerate(input_files):
        img_name = os.path.basename(path)
        gt_path = os.path.join(opt.gt_folder, img_name)
        
        # Skip if GT doesn't exist
        if not os.path.exists(gt_path):
            print(f'Warning: GT not found for {img_name}, skipping...')
            continue
        
        print(f'Processing {idx + 1}/{total_files}: {img_name}')
        
        # Read input image
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).cuda()
        
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            
            # Padding to multiple of 16 (or 8)
            pad_size = opt.pad_size
            h_pad = (pad_size - (h_old % pad_size)) % pad_size
            w_pad = (pad_size - (w_old % pad_size)) % pad_size
            
            if h_pad != 0 or w_pad != 0:
                img_lq = torch.nn.functional.pad(img_lq, (0, w_pad, 0, h_pad), mode='reflect')
            
            # Inference
            start_time = time.time()
            img_clean = model(img_lq)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Remove padding
            img_clean = img_clean[..., :h_old, :w_old]
        
        # Convert to numpy and save
        output = img_clean.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
        output = (output * 255.0).round().astype(np.uint8)
        
        # Read GT image
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.uint8)
        
        # Calculate metrics
        psnr = calculate_psnr(gt_img, output, crop_border=0, test_y_channel=True)
        ssim = calculate_ssim(gt_img, output, crop_border=0, test_y_channel=True)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
        print(f'  PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}, Time: {processing_time:.4f}s')
        
        # Save output image if requested
        if opt.save_images:
            cv2.imwrite(os.path.join(opt.output_folder, img_name), output)
    
    # Print summary
    print('\n' + '='*60)
    print('TESTING COMPLETED')
    print('='*60)
    print(f'Total images processed: {len(psnr_list)}')
    print(f'Average PSNR: {np.mean(psnr_list):.4f} dB')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')
    print(f'PSNR std: {np.std(psnr_list):.4f} dB')
    print(f'SSIM std: {np.std(ssim_list):.4f}')
    print(f'Average processing time: {np.mean(processing_times):.4f}s per image')
    
    if opt.save_images:
        print(f'\nOutput images saved to: {opt.output_folder}')
    
    # Save metrics to file
    metrics_file = os.path.join(opt.output_folder, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('='*60 + '\n')
        f.write('HOGformer Test Results\n')
        f.write('='*60 + '\n\n')
        f.write(f'Checkpoint: {opt.ckpt_path}\n')
        f.write(f'Input folder: {opt.input_folder}\n')
        f.write(f'GT folder: {opt.gt_folder}\n')
        f.write(f'Total images: {len(psnr_list)}\n\n')
        
        f.write('-'*60 + '\n')
        f.write('Summary Statistics\n')
        f.write('-'*60 + '\n')
        f.write(f'Average PSNR: {np.mean(psnr_list):.4f} dB (std: {np.std(psnr_list):.4f})\n')
        f.write(f'Average SSIM: {np.mean(ssim_list):.4f} (std: {np.std(ssim_list):.4f})\n')
        f.write(f'Min PSNR: {np.min(psnr_list):.4f} dB\n')
        f.write(f'Max PSNR: {np.max(psnr_list):.4f} dB\n')
        f.write(f'Min SSIM: {np.min(ssim_list):.4f}\n')
        f.write(f'Max SSIM: {np.max(ssim_list):.4f}\n')
        f.write(f'Average processing time: {np.mean(processing_times):.4f}s\n\n')
        
        f.write('-'*60 + '\n')
        f.write('Per-Image Results\n')
        f.write('-'*60 + '\n')
        f.write(f'{"Image Name":<40} {"PSNR (dB)":<12} {"SSIM":<10} {"Time (s)":<10}\n')
        f.write('-'*60 + '\n')
        
        for i, path in enumerate(input_files[:len(psnr_list)]):
            img_name = os.path.basename(path)
            f.write(f'{img_name:<40} {psnr_list[i]:<12.4f} {ssim_list[i]:<10.4f} {processing_times[i]:<10.4f}\n')
    
    print(f'Metrics saved to: {metrics_file}')
    print('='*60)


def set_seed(seed=100):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test HOGformer on image restoration tasks')
    
    parser.add_argument('--ckpt_path', type=str, required=True, 
                        help='Path to the model checkpoint')
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Path to the input (degraded) images folder')
    parser.add_argument('--gt_folder', type=str, required=True, 
                        help='Path to the ground truth images folder')
    parser.add_argument('--output_folder', type=str, default='./results', 
                        help='Path to save output images and metrics')
    parser.add_argument('--save_images', action='store_true', 
                        help='Save restored images')
    parser.add_argument('--pad_size', type=int, default=16, 
                        help='Padding size (8 or 16)')
    parser.add_argument('--seed', type=int, default=100, 
                        help='Random seed for reproducibility')
    
    opt = parser.parse_args()
    
    # Set random seed
    set_seed(opt.seed)
    
    # Handle existing output folder
    if os.path.exists(opt.output_folder):
        new_name = opt.output_folder.rstrip('/') + '_' + time.strftime("%Y%m%d%H%M%S")
        shutil.move(opt.output_folder, new_name)
        print(f"Output folder already exists. Renamed to {new_name}")
    
    os.makedirs(opt.output_folder, exist_ok=True)
    print(f"Output folder created: {opt.output_folder}")
    
    # Run testing
    main()