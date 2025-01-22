# Nail Diseases Detection System - Graduation Project

![Project Logo](https://github.com/user-attachments/assets/ee9e3cda-dd2f-4bc3-ae3f-1ceddf40ba45)

Welcome to the **Nail Diseases Detection System** repository! This project is a web-based application that uses **image processing** and **deep learning** techniques to detect and classify various nail diseases. The goal is to provide an accessible and accurate tool for early detection of nail diseases, helping both doctors and the general public.

---

## Table of Contents
1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [How It Works](#how-it-works)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Results](#results)
10. [Future Work](#future-work)
11. [Contributing](#contributing)
12. [Acknowledgments](#acknowledgments)
13. [License](#license)

---

## Overview
This project is a **Graduation Project** developed by students from **Ahram Canadian University, Faculty of Computer Science & Information Technology**. The system uses **Convolutional Neural Networks (CNNs)** and **image processing** techniques to detect and classify nail diseases from images. The application is designed to be user-friendly, allowing users to upload images of their nails and receive a diagnosis with high accuracy.

---

## Motivation
Nail diseases can be indicators of underlying health issues, and early detection is crucial for effective treatment. However, diagnosing nail diseases can be time-consuming and requires specialized knowledge. This project aims to:
- Provide a **quick and accurate** tool for nail disease detection.
- Assist doctors in diagnosing diseases more efficiently.
- Help the general public identify potential nail diseases and seek appropriate medical attention.

---

## Features
- **User-Friendly Interface**: Easy-to-use web interface for uploading nail images and receiving results.
- **High Accuracy**: The model achieves an accuracy of **88.90%** using **ResNet-50** architecture.
- **Multiple Disease Detection**: The system can detect **9 different nail diseases**.
- **Data Augmentation**: Techniques like rotation, cropping, and flipping are used to improve model performance.
- **Transfer Learning**: Pre-trained models like **VGG-16** and **ResNet-50** are used to enhance accuracy.

---

## Technologies Used
- **Programming Languages**: Python, HTML5, CSS3, JavaScript
- **Frameworks**: TensorFlow, PyTorch, Keras
- **Libraries**: OpenCV, NumPy, Pandas, Scikit-learn
- **Web Development**: Bootstrap, Flask (for backend)
- **Data Augmentation**: Image rotation, cropping, zooming, flipping
- **Model Training**: AdamW optimizer, Dropout, Regularization, Early Stopping

---

## Dataset
The dataset used in this project contains **10 types of nail diseases** with a total of **12700 raw images**. The diseases included are:
1. **Healthy Nail**
2. **Terry's Nail**
3. **Clubbing**
4. **Acral Lentiginous Melanoma**
5. **Pitting**
6. **Blue Finger**
7. **Koilonychia**
8. **Beau's Lines**
9. **Muehrcke's Lines**

---

## How It Works
1. **Image Upload**: Users upload an image of their nail through the web interface.
2. **Preprocessing**: The image is preprocessed (blurring, shearing, equalization) to prepare it for the model.
3. **Model Prediction**: The preprocessed image is fed into the **CNN model**, which predicts the disease.
4. **Result Display**: The system displays the predicted disease along with the probability of each possible disease.
5. **Recommendation**: Users are advised on which type of doctor to consult based on the diagnosis.

---

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/FathyHesham/Nail-Diseases-Detection-System-Graduation-Project.git
   cd Nail-Diseases-Detection-System-Graduation-Project
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**:
   Open your browser and go to `http://127.0.0.1:5000/` to use the application.

---

## Usage
1. **Home Page**: Start by clicking the "Start" button.

![image](https://github.com/user-attachments/assets/3c73fa07-7491-46cf-8999-feeedeec5b79)

2. **Notes Page**: Follow the instructions on how to take a clear picture of your nail.

![image](https://github.com/user-attachments/assets/b8a68d04-015b-4f93-883a-605eec77ab31)

3. **Form Page**: Fill out the form and upload your nail image.

![image](https://github.com/user-attachments/assets/b19bce06-4d4a-49d0-8279-2ea8988fa1f6)

4. **Prediction Page**: The system will process the image and display the results.

![image](https://github.com/user-attachments/assets/3c04fb53-6003-488f-9336-06793f72a0aa)

5. **Results Page**: View the diagnosis and recommendations.

![image](https://github.com/user-attachments/assets/edf1948c-0bae-45b0-a0b2-040d06f74582)

6. **Search Page**: Search for previous results using your unique ID.

![image](https://github.com/user-attachments/assets/46b2dcb9-6e23-49db-b0e7-7ac4e7b1a45a)

7. **Contact Page**: Contact the support team for any inquiries.

![image](https://github.com/user-attachments/assets/a2d6d6c4-568a-4879-9b8a-7ea3cca28ac0)

---

## Results
The final model achieved an accuracy of **88.90%** using the **ResNet** architecture. Below are some sample results:

![Prediction Results](https://github.com/user-attachments/assets/f2e71d1f-a885-4416-8518-9d404082dd06)

---

## Future Work
1. **Expand Dataset**: Collect more data for additional nail diseases to improve the model's accuracy and coverage.
2. **Increase Accuracy**: Continue refining the model to achieve higher accuracy.
3. **Platform Expansion**: Develop mobile applications for **iOS** and **Android** to make the tool more accessible.
4. **Healthcare Integration**: Collaborate with healthcare organizations to integrate the tool into their systems for wider use.

---

## Contributing
We welcome contributions to this project! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

## Acknowledgments
We would like to thank:
- **Dr. Sherif Eletriby** for his guidance and support throughout this project.
- **Ahram Canadian University** and the **Faculty of Computer Science & Information Technology** for providing the resources and knowledge to complete this project.
- Our families and friends for their endless support and encouragement.

---

## License
This project is protected under intellectual property rights and is the sole property of **Ahram Canadian University**. Any use or reproduction of this work requires permission from the university and the respective supervisor.

---

## Contact
For any inquiries or further information, feel free to reach out:

- **Mail**: [fathyhesham2001@gmail.com](mailto:fathyhesham2001@gmail.com)
- **LinkedIn**: [Fathy Hesham Fathy](https://www.linkedin.com/in/fathy-hesham-fathy/)

---

**Thank you for visiting our project!** 🚀
