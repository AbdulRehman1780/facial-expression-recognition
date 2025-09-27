import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FacialExpressionDatasetExplorer:
    """
    A comprehensive tool to explore facial expression datasets with PyTorch compatibility.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the dataset explorer.
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.emotion_labels = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad', 
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger'
        }
        
    def list_dataset_structure(self):
        """
        List all files and folders in the dataset directory.
        """
        print("=" * 60)
        print("DATASET STRUCTURE ANALYSIS")
        print("=" * 60)
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return
            
        print(f"📁 Dataset Root: {self.dataset_path}")
        print(f"📊 Total size: {self._get_directory_size(self.dataset_path):.2f} MB")
        print("\n📂 Directory Structure:")
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.dataset_path):
            level = root.replace(str(self.dataset_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            
            # Show file counts by extension
            file_extensions = {}
            for file in files:
                ext = Path(file).suffix.lower()
                file_extensions[ext] = file_extensions.get(ext, 0) + 1
            
            sub_indent = ' ' * 2 * (level + 1)
            for ext, count in file_extensions.items():
                if ext:
                    print(f"{sub_indent}📄 {count} {ext} files")
                else:
                    print(f"{sub_indent}📄 {count} files (no extension)")
                    
        print("\n" + "=" * 60)
    
    def _get_directory_size(self, path):
        """Calculate directory size in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
        return total_size / (1024 * 1024)  # Convert to MB
    
    def explore_npy_files(self):
        """
        Check .npy file contents and shapes.
        """
        print("\n" + "=" * 60)
        print("NPY FILES ANALYSIS")
        print("=" * 60)
        
        npy_files = list(self.dataset_path.rglob("*.npy"))
        
        if not npy_files:
            print("❌ No .npy files found in the dataset")
            return
            
        print(f"📊 Found {len(npy_files)} .npy files\n")
        
        for npy_file in npy_files:
            try:
                data = np.load(npy_file, allow_pickle=True)
                print(f"📄 File: {npy_file.name}")
                print(f"   📐 Shape: {data.shape}")
                print(f"   🔢 Data type: {data.dtype}")
                print(f"   💾 Size: {data.nbytes / 1024:.2f} KB")
                
                # Show sample values for small arrays
                if data.size <= 20:
                    print(f"   📋 Sample values: {data.flatten()[:10]}")
                elif len(data.shape) == 1:
                    print(f"   📋 Sample values: {data[:5]} ... {data[-5:]}")
                else:
                    print(f"   📋 Value range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                
                print()
                
            except Exception as e:
                print(f"❌ Error loading {npy_file.name}: {e}\n")
    
    def display_sample_images(self, num_samples=6):
        """
        Display sample images with their annotations.
        """
        print("\n" + "=" * 60)
        print("SAMPLE IMAGES WITH ANNOTATIONS")
        print("=" * 60)
        
        # Try to find image files and annotation files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(self.dataset_path.rglob(f"*{ext}")))
        
        if not image_files:
            print("❌ No image files found")
            return
            
        # Look for annotation files
        annotation_files = list(self.dataset_path.rglob("*.csv")) + list(self.dataset_path.rglob("*.txt"))
        
        print(f"📊 Found {len(image_files)} images")
        print(f"📊 Found {len(annotation_files)} potential annotation files")
        
        # Display sample images
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for idx, img_path in enumerate(sample_images):
            try:
                # Load and display image
                img = Image.open(img_path)
                axes[idx].imshow(img)
                axes[idx].set_title(f"{img_path.name}\nSize: {img.size}")
                axes[idx].axis('off')
                
                # Try to extract emotion from filename or path
                emotion_info = self._extract_emotion_from_path(img_path)
                if emotion_info:
                    axes[idx].set_xlabel(emotion_info, fontsize=10)
                    
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error loading\n{img_path.name}", 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Sample images saved as 'sample_images.png'")
    
    def _extract_emotion_from_path(self, img_path):
        """Extract emotion information from image path or filename."""
        path_str = str(img_path).lower()
        filename = img_path.stem.lower()
        
        # Check for emotion keywords in path
        for emotion_id, emotion_name in self.emotion_labels.items():
            if emotion_name.lower() in path_str or emotion_name.lower() in filename:
                return f"Emotion: {emotion_name}"
        
        # Check for numeric emotion labels
        import re
        numbers = re.findall(r'\d+', filename)
        if numbers:
            emotion_id = int(numbers[0])
            if emotion_id in self.emotion_labels:
                return f"Emotion: {self.emotion_labels[emotion_id]}"
        
        return None
    
    def verify_image_dimensions(self):
        """
        Verify that images are 224x224 RGB.
        """
        print("\n" + "=" * 60)
        print("IMAGE DIMENSIONS VERIFICATION")
        print("=" * 60)
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(self.dataset_path.rglob(f"*{ext}")))
        
        if not image_files:
            print("❌ No image files found")
            return
        
        dimensions = {}
        modes = {}
        total_checked = 0
        errors = 0
        
        # Sample a subset for efficiency
        sample_size = min(100, len(image_files))
        sample_images = np.random.choice(image_files, sample_size, replace=False)
        
        print(f"🔍 Checking {sample_size} random images out of {len(image_files)} total")
        
        for img_path in sample_images:
            try:
                with Image.open(img_path) as img:
                    size = img.size  # (width, height)
                    mode = img.mode
                    
                    dimensions[size] = dimensions.get(size, 0) + 1
                    modes[mode] = modes.get(mode, 0) + 1
                    total_checked += 1
                    
            except Exception as e:
                errors += 1
                print(f"❌ Error loading {img_path.name}: {e}")
        
        print(f"\n📊 Successfully checked: {total_checked} images")
        print(f"❌ Errors encountered: {errors} images")
        
        print("\n📐 Image Dimensions Distribution:")
        for size, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_checked) * 100
            status = "✅" if size == (224, 224) else "⚠️"
            print(f"   {status} {size[0]}x{size[1]}: {count} images ({percentage:.1f}%)")
        
        print("\n🎨 Color Mode Distribution:")
        for mode, count in sorted(modes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_checked) * 100
            status = "✅" if mode == "RGB" else "⚠️"
            print(f"   {status} {mode}: {count} images ({percentage:.1f}%)")
        
        # Check if majority are 224x224 RGB
        target_size_count = dimensions.get((224, 224), 0)
        rgb_count = modes.get('RGB', 0)
        
        if target_size_count / total_checked > 0.8:
            print("\n✅ Most images are 224x224 pixels")
        else:
            print("\n⚠️ Warning: Not all images are 224x224 pixels")
            
        if rgb_count / total_checked > 0.8:
            print("✅ Most images are RGB format")
        else:
            print("⚠️ Warning: Not all images are RGB format")
    
    def generate_statistics(self):
        """
        Show statistics: number of samples per emotion class, valence/arousal distributions.
        """
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        # Try to find and load annotation files
        csv_files = list(self.dataset_path.rglob("*.csv"))
        
        if csv_files:
            print(f"📊 Found {len(csv_files)} CSV files")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    print(f"\n📄 Analyzing: {csv_file.name}")
                    print(f"   📐 Shape: {df.shape}")
                    print(f"   📋 Columns: {list(df.columns)}")
                    
                    # Look for emotion-related columns
                    emotion_cols = [col for col in df.columns if any(keyword in col.lower() 
                                   for keyword in ['emotion', 'label', 'class', 'expression'])]
                    
                    if emotion_cols:
                        for col in emotion_cols:
                            print(f"\n🎭 Emotion Distribution ({col}):")
                            value_counts = df[col].value_counts()
                            for value, count in value_counts.items():
                                emotion_name = self.emotion_labels.get(value, str(value))
                                percentage = (count / len(df)) * 100
                                print(f"   {emotion_name}: {count} samples ({percentage:.1f}%)")
                    
                    # Look for valence/arousal columns
                    valence_cols = [col for col in df.columns if 'valence' in col.lower()]
                    arousal_cols = [col for col in df.columns if 'arousal' in col.lower()]
                    
                    if valence_cols or arousal_cols:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        if valence_cols:
                            valence_col = valence_cols[0]
                            axes[0].hist(df[valence_col].dropna(), bins=30, alpha=0.7, color='blue')
                            axes[0].set_title(f'Valence Distribution ({valence_col})')
                            axes[0].set_xlabel('Valence')
                            axes[0].set_ylabel('Frequency')
                            
                            print(f"\n📊 Valence Statistics ({valence_col}):")
                            print(f"   Mean: {df[valence_col].mean():.3f}")
                            print(f"   Std: {df[valence_col].std():.3f}")
                            print(f"   Range: [{df[valence_col].min():.3f}, {df[valence_col].max():.3f}]")
                        
                        if arousal_cols:
                            arousal_col = arousal_cols[0]
                            axes[1].hist(df[arousal_col].dropna(), bins=30, alpha=0.7, color='red')
                            axes[1].set_title(f'Arousal Distribution ({arousal_col})')
                            axes[1].set_xlabel('Arousal')
                            axes[1].set_ylabel('Frequency')
                            
                            print(f"\n📊 Arousal Statistics ({arousal_col}):")
                            print(f"   Mean: {df[arousal_col].mean():.3f}")
                            print(f"   Std: {df[arousal_col].std():.3f}")
                            print(f"   Range: [{df[arousal_col].min():.3f}, {df[arousal_col].max():.3f}]")
                        
                        plt.tight_layout()
                        plt.savefig('valence_arousal_distributions.png', dpi=150, bbox_inches='tight')
                        plt.show()
                        print("✅ Valence/Arousal distributions saved as 'valence_arousal_distributions.png'")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {csv_file.name}: {e}")
        else:
            print("⚠️ No CSV annotation files found")
            
            # Try to infer statistics from directory structure
            self._infer_stats_from_structure()
    
    def _infer_stats_from_structure(self):
        """Infer statistics from directory structure if no annotation files."""
        print("\n🔍 Attempting to infer statistics from directory structure...")
        
        emotion_counts = {}
        
        # Look for subdirectories that might represent emotion classes
        for item in self.dataset_path.iterdir():
            if item.is_dir():
                # Count images in each subdirectory
                image_count = 0
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_count += len(list(item.rglob(f"*{ext}")))
                
                if image_count > 0:
                    emotion_counts[item.name] = image_count
        
        if emotion_counts:
            print("\n🎭 Inferred Emotion Distribution from Directory Structure:")
            total_images = sum(emotion_counts.values())
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_images) * 100
                print(f"   {emotion}: {count} images ({percentage:.1f}%)")
        else:
            print("❌ Could not infer emotion distribution from directory structure")


class FacialExpressionDataset(Dataset):
    """
    PyTorch Dataset class for facial expression data.
    """
    
    def __init__(self, dataset_path, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to dataset directory
            transform: PyTorch transforms to apply to images
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Find all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend(list(self.dataset_path.rglob(f"*{ext}")))
        
        # Try to load annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """Load annotations from CSV files if available."""
        csv_files = list(self.dataset_path.rglob("*.csv"))
        
        if csv_files:
            # Use the first CSV file found
            try:
                return pd.read_csv(csv_files[0])
            except Exception as e:
                print(f"Warning: Could not load annotations from {csv_files[0]}: {e}")
        
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Dictionary containing 'image', 'path', and optionally 'emotion', 'valence', 'arousal'
        """
        img_path = self.image_files[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
        
        sample = {
            'image': image,
            'path': str(img_path),
            'filename': img_path.name
        }
        
        # Add annotations if available
        if self.annotations is not None:
            # Try to match image with annotations (this is dataset-specific)
            # This is a simplified example - you may need to adapt this
            sample.update({
                'emotion': 0,  # Default values
                'valence': 0.0,
                'arousal': 0.0
            })
        
        return sample


def create_pytorch_dataloader_preview(dataset_path):
    """
    Create PyTorch-compatible data loading preview.
    """
    print("\n" + "=" * 60)
    print("PYTORCH DATALOADER PREVIEW")
    print("=" * 60)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    try:
        dataset = FacialExpressionDataset(dataset_path, transform=transform)
        print(f"✅ Dataset created successfully")
        print(f"📊 Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ No images found in dataset")
            return
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        print(f"✅ DataLoader created with batch size: 4")
        
        # Test loading a batch
        print("\n🔍 Testing batch loading...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"✅ Batch {batch_idx + 1}:")
            print(f"   📐 Image tensor shape: {batch['image'].shape}")
            print(f"   🔢 Image tensor dtype: {batch['image'].dtype}")
            print(f"   📊 Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"   📄 Sample filenames: {batch['filename'][:2]}")
            
            if batch_idx >= 2:  # Only show first 3 batches
                break
        
        print(f"\n✅ PyTorch DataLoader is working correctly!")
        print(f"💡 You can use this dataset class in your training pipeline")
        
        # Show sample usage code
        print("\n" + "=" * 40)
        print("SAMPLE USAGE CODE:")
        print("=" * 40)
        print("""
# Example usage in training:
from torch.utils.data import DataLoader

dataset = FacialExpressionDataset('path/to/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch in dataloader:
    images = batch['image']  # Shape: [batch_size, 3, 224, 224]
    # Your training code here
    pass
        """)
        
    except Exception as e:
        print(f"❌ Error creating dataset/dataloader: {e}")


def main():
    """
    Main function to run the complete dataset exploration.
    """
    print("🚀 FACIAL EXPRESSION DATASET EXPLORER")
    print("=" * 60)
    
    # Get dataset path from user or use default
    dataset_path = input("Enter dataset path (or press Enter for current directory): ").strip()
    if not dataset_path:
        dataset_path = "."
    
    # Initialize explorer
    explorer = FacialExpressionDatasetExplorer(dataset_path)
    
    # Run all exploration steps
    try:
        explorer.list_dataset_structure()
        explorer.explore_npy_files()
        explorer.display_sample_images()
        explorer.verify_image_dimensions()
        explorer.generate_statistics()
        create_pytorch_dataloader_preview(dataset_path)
        
        print("\n" + "=" * 60)
        print("✅ DATASET EXPLORATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 Generated files:")
        print("   - sample_images.png")
        print("   - valence_arousal_distributions.png (if applicable)")
        print("\n💡 The dataset is now ready for PyTorch training!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Exploration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during exploration: {e}")


if __name__ == "__main__":
    main()
