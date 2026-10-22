# 🚗 Vehicle Damage Detection

A Streamlit web app that classifies the damage on a car from a photo. Upload an image, click **Predict**, and the app shows the predicted damage class with a confidence score.

The model is a ResNet50 fine-tuned with PyTorch using transfer learning.

## Classes

The model predicts one of six classes:

| Class |
|---|
| Front Breakage |
| Front Crushed |
| Front Normal |
| Rear Breakage |
| Rear Crushed |
| Rear Normal |

## Features

- Upload a car image (`.jpg`, `.jpeg`, `.png`) and preview it
- One-click prediction with the predicted class and confidence
- Model loads once and is reused for every prediction
- Runs on CPU or GPU automatically
- Clear error messages for a missing model file or an invalid image

## Project structure

```
Car-Damage-Detection/
├── README.md
├── .gitignore
├── dataset/                        # Training images, one sub-folder per class (not in the repo)
├── streamlit_app/
│   ├── model/
│   │   └── saved_model.pth         # Trained weights
│   ├── app.py                      # Streamlit UI
│   ├── model_helper.py             # Model definition, preprocessing, prediction
│   └── requirement.txt             # Python dependencies
├── car_damage_prediction.ipynb     # Model training and evaluation
└── hyper_parameter_tunung.ipynb    # Hyperparameter tuning with Optuna
```

The `dataset/` folder is not uploaded to GitHub because of its size. It is only needed if you want to re-train the model.

## Setup

1. **Open a terminal in the app folder:**

   ```bash
   cd streamlit_app
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate
   # Mac / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirement.txt
   ```

4. **Add the trained model.** Running the training notebook ends with:

   ```python
   torch.save(model.state_dict(), 'saved_model.pth')
   ```

   This saves the file next to the notebook. The repository already includes it in `streamlit_app/model/`; if you re-train, move the new file there so the app can find it. The file must contain the **ResNet50** weights (the `CarClassifierResNet` model). Weights from another architecture will fail to load.

## Run the app

From inside the `streamlit_app` folder:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Training the model

Open `car_damage_prediction.ipynb` and run the cells in order. The notebook reads images from `./dataset` using `ImageFolder`, so you need to add your own `dataset/` folder next to the notebook, containing one sub-folder per class (the six class names above). It trains and compares a simple CNN, a CNN with regularization, EfficientNet-B0 and ResNet50, then saves the final model.

`hyper_parameter_tunung.ipynb` contains the Optuna search used to tune the learning rate and dropout rate.

## How it works

1. The uploaded image is converted to RGB and resized to 224×224.
2. It is normalized with the ImageNet mean and standard deviation, the same preprocessing used in training (without the random augmentations).
3. The ResNet50 model outputs scores for the six classes, and softmax converts them into probabilities.
4. The class with the highest probability is shown along with its confidence.

## Model details

- **Base model:** ResNet50 pretrained on ImageNet
- **Fine-tuning:** all layers frozen except `layer4`, plus a new classifier head (`Dropout` followed by `Linear` with 6 outputs)
- **Input size:** 224×224
- **Loss and optimizer:** CrossEntropyLoss with Adam (weight decay 1e-4)
- **Augmentation (training only):** random horizontal flip, rotation (±10°), brightness, contrast and saturation jitter
- **Hyperparameters:** learning rate and dropout rate tuned with Optuna
- **Validation accuracy:** roughly 76% on a held-out set of 575 images

Several models were compared during development, from a simple CNN (about 54%) and EfficientNet-B0 (about 65%) to the fine-tuned ResNet50 (about 76%).

## Deployment notes

- On Streamlit Community Cloud, set the main file path to `streamlit_app/app.py`.
- Hosting services such as Streamlit Community Cloud look for a file named `requirements.txt` (with an "s"). Rename `requirement.txt` before deploying.
- For Linux hosting, use the CPU wheel index for PyTorch to avoid a very large download (add `--extra-index-url https://download.pytorch.org/whl/cpu` at the top of the requirements file).
- `saved_model.pth` is about 95 MB, which is close to GitHub's 100 MB file limit. Use Git LFS or host the weights elsewhere.

## Tech stack

Python, PyTorch, torchvision, Streamlit, Pillow, Optuna (for tuning).