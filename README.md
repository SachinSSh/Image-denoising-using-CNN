# Image Denoising Using CNN 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A deep learning-based solution for image denoising using Convolutional Neural Networks (CNNs), designed for the **RWPO** (Replace With Project/Organization Name) use case. This project aims to remove noise from images while preserving critical details, making it suitable for applications in medical imaging, photography, or satellite imagery.

## Results
### Example Denoising Comparison

| Noisy Input | Denoised Output |
|-------------|-----------------|
| ![Gaussian Noise](docs/gaussian_noise.png) | ![Denoised](docs/gaussian_denoised.png) |
| ![Salt-and-Pepper Noise](docs/salt_pepper_noise.png) | ![Denoised](docs/salt_pepper_denoised.png) |

**Quantitative Metrics** (test dataset):
| Noise Type          | PSNR (dB) | SSIM  |
|---------------------|-----------|-------|
| Gaussian (σ=25)     | 32.6      | 0.91  |
| Salt-and-Pepper (5%)| 34.1      | 0.89  |
| Poisson             | 30.8      | 0.85  |

*Note: Replace sample images in `docs/` with your own results.*

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features
- **CNN Architecture**: Utilizes a custom deep CNN model optimized for noise reduction.
- **Multiple Noise Types**: Handles Gaussian, salt-and-pepper, and Poisson noise.
- **Pretrained Models**: Includes pre-trained models for quick inference.
- **Metrics**: Evaluates performance using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
- **Easy Integration**: Simple API and CLI for training and inference.
- **Customizable**: Adjust hyperparameters, noise levels, and model depth.

## Installation

### Prerequisites
- Python 3.6+
- TensorFlow 2.x / Keras
- OpenCV
- scikit-image
- NumPy
- Matplotlib

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-denoising-using-cnn.git
   cd image-denoising-using-cnn
