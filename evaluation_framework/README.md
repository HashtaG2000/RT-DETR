# RT-DETR Evaluation Framework

**Author:** Ratnesh
**Purpose:** Evaluate RT-DETR models using the author's original code

## Folder Structure

```
evaluation_framework/
├── src/                    # Author's RT-DETR model code (DO NOT MODIFY)
├── configs/                # Author's model configurations (DO NOT MODIFY)
├── dataset/                # COCO validation dataset
│   └── coco/
│       ├── val2017/        # 5000 validation images
│       └── annotations/    # COCO annotations
├── outputs/                # Evaluation results will be saved here
└── RTDETR_Evaluation.ipynb # Main evaluation notebook (YOUR WORK)
```

## What This Is

This is a **clean evaluation framework** that:
- ✅ Uses the **author's original RT-DETR code** from `src/`
- ✅ Imports and calls author's functions directly
- ✅ Runs evaluation using author's methods
- ✅ Adds FPS benchmarking and visualization on top

## Quick Start

1. **Open the notebook:**
   ```bash
   jupyter notebook RTDETR_Evaluation.ipynb
   ```

2. **Run all cells** - The notebook will:
   - Load RT-DETR model using author's code
   - Run COCO evaluation
   - Measure FPS performance
   - Generate visualizations

## What's Inside

### Author's Code (src/)
- `src/core/` - Config loading (YAMLConfig)
- `src/nn/` - Model architectures
- `src/zoo/rtdetr/` - RT-DETR implementation
- `src/data/` - Data loading utilities

### Your Notebook (RTDETR_Evaluation.ipynb)
- Imports from author's `src/`
- Organizes evaluation workflow
- Adds FPS benchmarking
- Creates visualizations
- Generates summary reports

## Requirements

```bash
pip install torch torchvision pycocotools numpy matplotlib tqdm
```

## Notes

- **DO NOT modify** files in `src/` or `configs/` - these are the author's original code
- All your work should be in the Jupyter notebook
- Results are automatically saved to `outputs/`
