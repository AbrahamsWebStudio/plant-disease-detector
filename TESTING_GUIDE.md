# Testing Guide for Plant Disease Detector

This guide is written for another student who wants to test the project step by step.

## 1. Environment setup

1. Open a terminal.
2. Go to the project folder:
   ```bash
   cd /home/rafiki/plant_disease_detector
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Confirm that Python is available:
   ```bash
   python --version
   ```

## 2. Installing dependencies

If dependencies are not installed yet, run:

```bash
pip install -r requirements.txt
```

If pytest is needed for tests, run:

```bash
pip install pytest
```

## 3. Training the model

The training script is in the model folder.

Run:

```bash
python model/train_pipeline.py
```

Expected result:
- If the dataset is present, training should start.
- If the dataset folder is missing or empty, the script will stop with a clear error message.

## 4. Running the Django server

Go to the web folder:

```bash
cd web
python manage.py runserver
```

Open the URL in a browser:

```text
http://127.0.0.1:8000/
```

## 5. Uploading a valid image

1. Open the home page.
2. Choose an image file.
3. Click Predict.

Expected result:
- The app should show a prediction result page.
- The page should include a predicted class and a confidence score.

## 6. Uploading an invalid image

Try uploading:
- a text file
- an empty file
- a non-image file

Expected result:
- Django should reject the invalid upload or show an error message.

## 7. Testing each disease class

To test the model, use images that represent:
- Healthy leaves
- Early Blight leaves
- Late Blight leaves

Expected result:
- The app should return one of the supported class labels.
- The confidence value may vary depending on the image quality.

## 8. Expected outputs

When the app is working correctly, you should see:
- a home page with the upload form
- a result page after prediction
- a predicted class label
- a confidence score

## 9. Common errors and fixes

### Error: ModuleNotFoundError

Fix:
- Make sure the virtual environment is activated.
- Check that the project root is on the Python path.

### Error: Django migration or system check issue

Fix:
- Run:
  ```bash
  python manage.py check
  ```

### Error: Model file not found

Fix:
- Train the model first so the file is created in the model folder.

### Error: Invalid image upload

Fix:
- Upload a real image file in JPG or PNG format.

## Final checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Django server runs
- [ ] Home page opens
- [ ] Valid image uploads successfully
- [ ] Invalid image is handled properly
- [ ] Prediction result appears
