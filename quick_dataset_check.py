"""
Quick Dataset Check Script
A simplified version for immediate dataset exploration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

def quick_dataset_overview(dataset_path="."):
    """
    Perform a quick overview of the dataset structure and contents.
    """
    dataset_path = Path(dataset_path)
    
    print("🔍 QUICK DATASET OVERVIEW")
    print("=" * 50)
    print(f"📁 Dataset Path: {dataset_path.absolute()}")
    
    if not dataset_path.exists():
        print("❌ Dataset path does not exist!")
        return
    
    # 1. List directory structure
    print("\n📂 Directory Structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Count files by type
        file_types = {}
        for file in files:
            ext = Path(file).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        sub_indent = '  ' * (level + 1)
        for ext, count in file_types.items():
            if ext:
                print(f"{sub_indent}{count} {ext} files")
    
    # 2. Find and analyze .npy files
    print("\n📊 NPY Files:")
    npy_files = list(dataset_path.rglob("*.npy"))
    if npy_files:
        for npy_file in npy_files[:5]:  # Show first 5
            try:
                data = np.load(npy_file, allow_pickle=True)
                print(f"  {npy_file.name}: shape {data.shape}, dtype {data.dtype}")
            except Exception as e:
                print(f"  {npy_file.name}: Error - {e}")
    else:
        print("  No .npy files found")
    
    # 3. Find and check images
    print("\n🖼️ Images:")
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(dataset_path.rglob(f"*{ext}")))
    
    if image_files:
        print(f"  Found {len(image_files)} image files")
        
        # Check a few sample images
        sample_images = image_files[:5]
        dimensions = []
        modes = []
        
        for img_path in sample_images:
            try:
                with Image.open(img_path) as img:
                    dimensions.append(img.size)
                    modes.append(img.mode)
                    print(f"  {img_path.name}: {img.size}, {img.mode}")
            except Exception as e:
                print(f"  {img_path.name}: Error - {e}")
        
        # Check if images are 224x224 RGB
        if dimensions:
            target_size = (224, 224)
            correct_size = sum(1 for dim in dimensions if dim == target_size)
            rgb_mode = sum(1 for mode in modes if mode == 'RGB')
            
            print(f"  📐 {correct_size}/{len(dimensions)} images are 224x224")
            print(f"  🎨 {rgb_mode}/{len(dimensions)} images are RGB")
    else:
        print("  No image files found")
    
    # 4. Look for annotation files
    print("\n📋 Annotation Files:")
    csv_files = list(dataset_path.rglob("*.csv"))
    txt_files = list(dataset_path.rglob("*.txt"))
    
    if csv_files:
        print(f"  Found {len(csv_files)} CSV files:")
        for csv_file in csv_files[:3]:
            print(f"    {csv_file.name}")
    
    if txt_files:
        print(f"  Found {len(txt_files)} TXT files:")
        for txt_file in txt_files[:3]:
            print(f"    {txt_file.name}")
    
    if not csv_files and not txt_files:
        print("  No annotation files found")
    
    # 5. Test PyTorch compatibility
    print("\n🔧 PyTorch Compatibility Test:")
    try:
        # Create a simple transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        if image_files:
            # Test loading one image
            test_img = Image.open(image_files[0]).convert('RGB')
            tensor_img = transform(test_img)
            print(f"  ✅ Image tensor shape: {tensor_img.shape}")
            print(f"  ✅ Image tensor dtype: {tensor_img.dtype}")
            print(f"  ✅ PyTorch transforms working correctly")
        else:
            print("  ⚠️ No images to test PyTorch compatibility")
            
    except Exception as e:
        print(f"  ❌ PyTorch compatibility error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Quick overview completed!")


def display_sample_images(dataset_path=".", num_samples=4):
    """
    Display a few sample images from the dataset.
    """
    dataset_path = Path(dataset_path)
    
    # Find image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(dataset_path.rglob(f"*{ext}")))
    
    if not image_files:
        print("No images found to display")
        return
    
    # Select random samples
    import random
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create subplot
    fig, axes = plt.subplots(1, len(sample_files), figsize=(15, 4))
    if len(sample_files) == 1:
        axes = [axes]
    
    for idx, img_path in enumerate(sample_files):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_path.name}\n{img.size}")
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error\n{img_path.name}", 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample images saved as 'quick_sample_images.png'")


if __name__ == "__main__":
    # Get dataset path
    dataset_path = input("Enter dataset path (or press Enter for current directory): ").strip()
    if not dataset_path:
        dataset_path = "."
    
    # Run quick overview
    quick_dataset_overview(dataset_path)
    
    # Ask if user wants to see sample images
    show_images = input("\nDisplay sample images? (y/n): ").strip().lower()
    if show_images in ['y', 'yes']:
        display_sample_images(dataset_path)
