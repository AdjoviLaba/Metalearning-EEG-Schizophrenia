# Metalearning-EEG-Schizophrenia

## PArt 1 using MAML for EEG Classification of Schizophrenia

This repository implements **Model-Agnostic Meta-Learning (MAML)** for the classification of EEG data, specifically focused on distinguishing between healthy controls and schizophrenia patients. The project is implemented using **PyTorch** and provides a flexible structure for training and validating on spectrogram images derived from EEG data.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Validation and Testing](#validation-and-testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to apply **meta-learning** techniques (specifically MAML) to the classification of schizophrenia from EEG-derived spectrogram data. The core idea of MAML is to train a model in such a way that it can adapt to new tasks (e.g., new subjects) with only a few examples, which is ideal for medical datasets where data is often scarce.

Key features:
- **EEG Spectrograms**: The dataset consists of spectrogram images extracted from EEG recordings.
- **MAML Architecture**: A meta-learning algorithm that learns a model initialization, allowing the model to adapt quickly to new participants or conditions with few training examples.
- **Schizophrenia Diagnosis**: The project is focused on distinguishing schizophrenia patients from healthy controls based on EEG data.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/AdjoviLaba/Metalearning-EEG-Schizophrenia.git/
cd Metalearning-EEG-Schizophrenia
```

Install the required dependencies :


### Requirements
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Torchvision

If you're using Google Colab, you may not need to install some of these packages, as they are pre-installed in the Colab environment.

## Dataset

This project works with **EEG spectrogram images**. The images are organized by subject and stored in the following folder structure:

```
/data_directory/
    hc/  # Healthy Control Subjects
        hc01/
            image1.png
            image2.png
        hc02/
            image3.png
    sz/  # Schizophrenia Patients
        sz01/
            image1.png
            image2.png
```

To use your dataset, place the image files in the correct folder structure as shown above.

## Usage

### Preprocessing EEG Data

The EEG data should first be transformed into spectrogram images. You can use your preferred method for generating spectrograms from EEG signals. The generated images should be categorized into folders by subject (`hc` for healthy controls and `sz` for schizophrenia patients).

### Configuration

You can modify the `config.py` or set the parameters for training in `train_custom_dataset.py` to adjust hyperparameters, including:
- Number of adaptation steps
- Learning rate
- Batch size
- Number of epochs

## Training

To start training the MAML model, run the following command:

```bash
python train_custom_dataset.py --data_dir /path/to/data --output_dir /path/to/save/results
```

The training process will display the training and validation accuracy for each epoch. The model will be saved to the specified output directory after training.

## Validation and Testing

Validation and testing splits are performed using `sklearn.model_selection.train_test_split`. After training, you can evaluate the model on the test set by running:

```bash
python validate_model.py --data_dir /path/to/test_data --model_dir /path/to/saved_model
```

### Example for Testing and Validation:
You can assign subjects for validation and testing using the provided function in `train_custom_dataset.py`:

```python
# Split data into test and validation sets
from sklearn.model_selection import train_test_split
validate_hc, test_hc = train_test_split(hc_subject_ids, test_size=0.5)
validate_sz, test_sz = train_test_split(sz_subject_ids, test_size=0.5)
```

## Results

The results of training (including model checkpoints and training logs) will be saved in the output directory. The training and testing accuracies will be logged for analysis.

Sample output of training progress:

```
Epoch 0
Training accuracies:  [0.625, 0.95, 0.95, 0.95, 0.95]
Validation accuracy:  0.85
```

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your improvements.

1. Fork it
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

