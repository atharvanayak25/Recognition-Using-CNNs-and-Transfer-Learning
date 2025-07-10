# 🧠 Digit & Greek Letter Recognition using CNNs + Transfer Learning

This project explores deep learning techniques in computer vision using PyTorch. It begins with a handwritten digit recognition model trained on the MNIST dataset and extends to Greek letter classification via transfer learning. Additionally, the project includes CNN analysis, custom handwritten digit testing, and hyperparameter experimentation on FashionMNIST.

---

## 🚀 Features

- ✅ Convolutional Neural Network (CNN) trained from scratch on MNIST
- 🔁 Transfer learning to recognize Greek letters (α, β, γ, etc.)
- ✍️ Testing on custom handwritten digit inputs
- 🔬 Visualizations of learned filters and feature maps
- ⚙️ Extensive experiments on dropout, filters, batch size, and training epochs
- 📊 Graphs comparing training accuracy, loss, and training time

---

## 🗂️ Project Structure

PRCV_Project5/
├── src/ # Python scripts
│ ├── train_MNIST.py
│ ├── train_greek.py
│ ├── eval_MNIST.py
│ ├── test_handwritten.py
│ ├── examine_network.py
│ ├── greekletters_extension.py
│ ├── experiment_fashion.py
│ └── analyse_experiment.py
│
├── outputs/ 
│ ├── Task_1_Outputs/
│ ├── Task_2_Outputs/
│ ├── Task_3_Outputs/
│ ├── Task_4_Outputs/
│ ├── greekletters_extension_Outputs/
|
│
├── datasets 
├── README.md
├── .gitignore
├── AtharvaNayak_PRCV_Project5.pdf


---

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/atharvanayak25/Recognition-Using-CNNs-and-Transfer-Learning.git
cd Recognition-Using-CNNs-and-Transfer-Learning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 🧠 Model Training

To train a model:

```bash
python main.py --mode train --model resnet50 --epochs 10
```

To evaluate:

```bash
python main.py --mode eval --model resnet50
```

You can customize hyperparameters, datasets, and model architecture in the config or via command-line arguments.

## 📊 Results

Below are the performance metrics for different models used in the project:

| Model    | Accuracy | Notes             |
|----------|----------|-------------------|
| ResNet50 | 94.2%    | Transfer Learning |
| VGG16    | 92.5%    | Fine-tuned        |


## 📦 Requirements

See requirements.txt for full list. Common dependencies include:

-Python 3.8+
-PyTorch / TensorFlow
-NumPy
-Matplotlib
-scikit-learn

## 🙋‍♂️ Author
Atharva Nayak
