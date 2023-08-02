#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:59:50 2023

@author: adam
"""

import sys
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout
from collections import defaultdict

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.directory_path = ""
        self.model = None
        self.class_mapping = {'Dead': 0, 'Veg': 1, 'Div': 2, 'PreDiv': 3, 'Spore': 4, 'New': 5}
        self.class_names = {v: k for k, v in self.class_mapping.items()}

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.model_label = QLabel("Select Model:")
        self.model_btn = QPushButton("Browse")
        self.model_btn.clicked.connect(self.load_model)

        self.directory_label = QLabel("Select Directory:")
        self.directory_btn = QPushButton("Browse")
        self.directory_btn.clicked.connect(self.select_directory)

        self.classify_btn = QPushButton("Classify Images")
        self.classify_btn.clicked.connect(self.classify_images)

        self.result_label = QLabel("Results will be displayed here.")

        layout.addWidget(self.model_label)
        layout.addWidget(self.model_btn)
        layout.addWidget(self.directory_label)
        layout.addWidget(self.directory_btn)
        layout.addWidget(self.classify_btn)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle("Image Classifier")
        self.show()

    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if file_path:
            self.model_path = file_path
            self.model = tf.keras.models.load_model(self.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
            self.model_label.setText(f"Selected Model: {self.model_path}")

    def select_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if dir_path:
            self.directory_path = dir_path
            self.directory_label.setText(f"Selected Directory: {self.directory_path}")

    def classify_images(self):
        if not self.model or not self.directory_path:
            self.result_label.setText("Please select both a model and a directory.")
            return

        class_counts = defaultdict(int)
        for image_file in os.listdir(self.directory_path):
            image_path = os.path.join(self.directory_path, image_file)
            try:
                class_index, _ = self.classify_object(image_path)
                class_name = self.class_names[class_index]
                class_counts[class_name] += 1
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

        result_text = "Class Counts:\n"
        for class_name, count in class_counts.items():
            result_text += f"{self.class_mapping[class_name]}:{class_name}, Count: {count}\n"

        self.result_label.setText(result_text)

    def classify_object(self, image_path):
        image = self.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        class_index = np.argmax(predictions)
        return class_index, predictions

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # EfficientNet input size
        image = np.array(image) / 255.0
        return image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    sys.exit(app.exec_())
