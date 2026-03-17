
# Efficient Image Shadow Removal utilizing a ConvNeXt-based Fusion Network
![NTIRE 2026](https://img.shields.io/badge/NTIRE-2026-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

This repository contains the inference code and resources for the shadow removal model. Follow the instructions below to set up the environment, download the pre-trained weights, and process your images.

## 1. Cloning the Repository
To get started, clone this repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/Mrudul2006/SVNIT-shadow-removal.git
cd SVNIT-shadow-removal
```
This will create a local copy of the repository and navigate to the project directory.

## 2. Downloading and Extracting Model Weights

The pre-trained model weights are hosted on Google Drive. You need to download and extract them into the `weights` directory before running the inference script.

### Step 2.1: Download the Weights
Use `gdown` to download the zip file directly from the command line:

```bash
gdown --id 17N36tVUtVKv3NRlNmodBu9yu5kQtI0-g -O weights.zip
```
### Step 2.2: Extract the Weights

Once downloaded, extract the contents into the `weights` folder:
```bash
# For Linux / Mac
unzip weights.zip -d weights

# For Windows
# Use File Explorer to right-click and extract the zip file into the 'weights' folder.
```
## 3. Running the Model

After setting up the repository and extracting the weights, place your input images in the `test_images` folder and run the testing script:

```bash
# Linux / Mac
python ./test.py --test_dir ./test_images --output_dir ./results --model_path ./weights/best_model.pth

# Windows (Command Prompt)
python test.py --test_dir .\test_images --output_dir .\results --model_path .\weights\best_model.pth

# Windows (PowerShell)
python .\test.py --test_dir .\test_images --output_dir .\results --model_path .\weights\best_model.pth
```
The script will process the images and save the clean, shadow-free outputs in the results folder.

### Important Notes
Dependencies: Ensure you have Python 3.8+ installed along with `torch`, `torchvision`, and `tqdm`. You can install the required packages using `pip install -r requirements.txt`.

gdown Tool: You can install `gdown` via pip using `pip install gdown`.

Alternative Download: If the command line download fails or you prefer using a browser, manually download the `weights` from [this Google Drive link](https://drive.google.com/file/d/17N36tVUtVKv3NRlNmodBu9yu5kQtI0-g/view?usp=sharing). Extract the downloaded weights.zip file into the `weights` directory.
### Support
Following these steps will ensure all necessary files are set up correctly. If you encounter any issues or require further assistance, please feel free to reach out to the team!