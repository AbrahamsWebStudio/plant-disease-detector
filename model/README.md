# Model training notes

This folder contains the AI training pipeline for the plant disease detection project.

## Files
- train_pipeline.py: builds and trains the image classifier
- tests/test_data_loader.py: basic smoke tests for the training module

## How to run
```bash
cd /home/rafiki/plant_disease_detector
source venv/bin/activate
python model/train_pipeline.py
```

## Expected behavior
If the dataset is missing or empty, the script will stop with a clear error message.
