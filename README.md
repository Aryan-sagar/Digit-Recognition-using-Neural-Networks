#  Handwritten Digit Classifier

A professional-grade deep learning project that classifies handwritten digits (0–9) with high accuracy using Convolutional Neural Networks (CNNs). Designed for educational, demonstrative, and deployment purposes.

---

## 🔍 Overview

This project utilizes the MNIST dataset and leverages TensorFlow and Keras to train a robust CNN model capable of recognizing handwritten digits with high precision. It includes model training, evaluation, and an optional web-based deployment using Flask.

---

## 📌 Features

- Convolutional Neural Network (CNN) architecture
- Trained on the standard MNIST dataset
- Model achieves >98% accuracy
- Clean and modular codebase
- Optional Flask app for real-time digit prediction

---

## 🧰 Tech Stack

- **Language:** Python 3.x  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Flask (optional)  
- **Dataset:** MNIST  
- **Development Tools:** Jupyter Notebook, VS Code / PyCharm

---

## 💻 Installation

```bash
git clone https://github.com/yourusername/intelligent-digit-classifier.git
cd intelligent-digit-classifier
pip install -r requirements.txt
```

To run the notebook:
```bash
jupyter notebook DigitClassifier.ipynb
```

To run the Flask web app:
```bash
cd app/
python app.py
```

---

## 🧪 Model Architecture

```python
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

## 📊 Results

- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~98%
- **Inference Speed:** ~1 ms per image

---

## 📷 Screenshots

> *(Add UI screenshots and training graphs if available)*

---

## 🚀 Future Scope

- Enhance UI/UX of Flask app
- Train on a larger custom handwritten dataset
- Deploy on Heroku or Streamlit for demo access
- Implement CNN variants like ResNet or MobileNet

---

## 🤝 Contributions

Feel free to fork, clone, or contribute. Pull requests are welcome.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
