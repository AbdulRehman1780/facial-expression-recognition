# Facial Expression Analysis: Multi-Task PyTorch Pipeline

This repository provides a complete, end-to-end PyTorch pipeline for facial expression analysis with multi-task learning:
- 8-class emotion classification
- Valence/Arousal regression

It includes dataset utilities, dual-head ResNet and EfficientNet models, training with AMP, comprehensive evaluation metrics, visualization, model comparison, and report generation. It can be run locally or end-to-end on Google Colab.

## Repository Structure (Key Files)

- `main.py` — Single-file pipeline with CLI phases:
  - `data_exploration`: dataset overview and sanity checks
  - `train_models`: trains ResNet-50 and EfficientNet-B0 multi-task models
  - `evaluate_models`: computes all metrics and saves figures
  - `generate_report`: produces a concise markdown summary

- Dataset and utilities:
  - `facial_expression_dataset.py` — Full Dataset class, transforms, loaders
  - `dataset_explorer.py`, `quick_dataset_check.py` — Dataset exploration tools

- Models:
  - `multitask_resnet.py` — MultiTaskResNet (dual heads)
  - `multitask_efficientnet.py` — MultiTaskEfficientNet-B0 (dual heads)

- Training:
  - `train.py` — Single-architecture training script
  - `train_both.py` — Trains both architectures sequentially
  - `multitask_loss.py` — Combined loss with weighting and scheduling

- Evaluation and analysis:
  - `evaluate.py` — All PDF-required metrics (classification + continuous)
  - `results_analysis.py` — Curves, confusion matrices, VA scatter, params/timing
  - `model_comparison.py` — Accuracy/time comparison + significance testing
  - `qualitative_examples.py` — Grids of correct/incorrect predictions

- Reporting:
  - `report_generator.py` — Markdown content generator for the PDF report
  - `short_report.py` — Concise, PDF-ready summary from existing outputs
  - `requirements.txt` — Dependencies

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run everything from the single-file pipeline
- Data exploration
```bash
python main.py --phase data_exploration --data_root path/to/Dataset
```

- Train both models (ResNet-50 and EfficientNet-B0)
```bash
python main.py --phase train_models --data_root path/to/Dataset --epochs 50 --batch_size 32 --lr 1e-3
```

- Evaluate models (metrics + plots)
```bash
python main.py --phase evaluate_models \
  --data_root path/to/Dataset \
  --resnet_ckpt checkpoints_both/<timestamp>/resnet50/best.pth \
  --effnet_ckpt checkpoints_both/<timestamp>/efficientnet_b0/best.pth \
  --batch_size 64
```

- Generate short report (markdown)
```bash
python main.py --phase generate_report \
  --data_root path/to/Dataset \
  --resnet_ckpt checkpoints_both/<timestamp>/resnet50/best.pth \
  --effnet_ckpt checkpoints_both/<timestamp>/efficientnet_b0/best.pth \
  --report_out report_outputs
```

### Google Colab
- Upload the project to `/content/DL_a2` and dataset to `/content/DL_a2/Dataset`.
- Install dependencies and run the same `main.py` commands.
- Mixed precision (AMP) and GPU are used automatically when available.

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
