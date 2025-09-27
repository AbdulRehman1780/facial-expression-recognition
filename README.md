# Facial Expression Dataset Explorer

This repository contains Python scripts to explore and analyze facial expression datasets for deep learning projects using PyTorch.

## Files Overview

### 1. `dataset_explorer.py` - Comprehensive Dataset Analysis
The main script that provides detailed analysis of your facial expression dataset including:
- Complete directory structure listing
- NPY file content analysis
- Sample image visualization with annotations
- Image dimension verification (224x224 RGB)
- Dataset statistics (emotion distribution, valence/arousal)
- PyTorch-compatible data loading preview

### 2. `quick_dataset_check.py` - Quick Overview
A simplified script for rapid dataset inspection:
- Basic directory structure
- File type counts
- Sample image properties
- PyTorch compatibility test

### 3. `requirements.txt` - Dependencies
All required Python packages for running the scripts.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Comprehensive Analysis
Run the full dataset explorer:
```bash
python dataset_explorer.py
```

When prompted, enter your dataset path or press Enter to use the current directory.

### Option 2: Quick Check
For a rapid overview:
```bash
python quick_dataset_check.py
```

## Features

### Dataset Structure Analysis
- Lists all files and folders recursively
- Shows file counts by extension
- Calculates total dataset size

### NPY File Analysis
- Loads and inspects .npy files
- Shows shapes, data types, and sample values
- Handles various numpy array formats

### Image Analysis
- Verifies image dimensions (checks for 224x224)
- Confirms RGB color mode
- Displays sample images with metadata
- Extracts emotion labels from filenames/paths

### Statistical Analysis
- Emotion class distribution
- Valence and arousal distributions (if available)
- Generates visualization plots
- Saves analysis results as PNG files

### PyTorch Integration
- Custom Dataset class implementation
- DataLoader compatibility testing
- Proper image transformations
- Batch loading verification

## Expected Dataset Structure

The scripts work with various dataset structures:

```
dataset/
├── images/
│   ├── emotion_0/
│   ├── emotion_1/
│   └── ...
├── annotations.csv
├── labels.npy
└── metadata/
```

Or:

```
dataset/
├── train/
│   ├── happy/
│   ├── sad/
│   └── ...
├── test/
└── annotations.csv
```

## Output Files

The scripts generate several output files:
- `sample_images.png` - Grid of sample images
- `valence_arousal_distributions.png` - Statistical plots
- `quick_sample_images.png` - Quick overview images

## Dataset Class Usage

After running the explorer, you can use the PyTorch Dataset class in your training:

```python
from dataset_explorer import FacialExpressionDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = FacialExpressionDataset('path/to/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Use in training loop
for batch in dataloader:
    images = batch['image']  # Shape: [batch_size, 3, 224, 224]
    filenames = batch['filename']
    # Your training code here
```

## Supported File Formats

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

### Annotations
- CSV files (.csv)
- NumPy arrays (.npy)
- Text files (.txt)

## Error Handling

The scripts include comprehensive error handling:
- Graceful handling of corrupted images
- Missing file warnings
- Invalid data format notifications
- Fallback options for incomplete datasets

## Customization

You can modify the scripts to:
- Add support for additional file formats
- Customize emotion label mappings
- Adjust visualization parameters
- Extend statistical analysis

## Troubleshooting

### Common Issues

1. **"No images found"**: Check that your dataset path is correct and contains image files
2. **"Permission denied"**: Ensure you have read access to the dataset directory
3. **"Module not found"**: Install missing dependencies using `pip install -r requirements.txt`
4. **Memory errors**: For large datasets, the scripts sample subsets to avoid memory issues

### Performance Tips

- For very large datasets, the scripts automatically sample subsets
- Use the quick check script first for initial validation
- Close matplotlib windows to free memory between runs

## Contributing

Feel free to extend these scripts for your specific dataset requirements. The modular design makes it easy to add new analysis features.

## License

This code is provided for educational and research purposes.
