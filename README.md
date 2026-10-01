# Code

This folder contains the code used for the two main stages of the methodology: **Activation Steering** and **SAE-Based Feature Analysis**.

## 1. Activation Steering

The `Activation Steering` folder contains the scripts used to identify the hallucination direction, determine the steering configuration, and perform the final evaluation.

* **`extraction.py`**
  Extracts the **hallucination direction** from the model activations.

* **`projection.py`**
  Computes the projections of different layers along the extracted hallucination direction. These results are used to determine an appropriate **projection threshold** for steering.

* **`configuration_selection.py`**
  Evaluates different **steering coefficients** to determine the steering configuration used for the final experiments.

* **`testing.py`**
  Performs the final testing. The hyperparameters obtained from the previous steps — the **hallucination direction, projection threshold, and steering coefficient** — are used during inference to generate and evaluate the steered responses.

The scripts therefore broadly follow the order:

extraction.py
      ↓
projection.py
      ↓
configuration_selection.py
      ↓
testing.py

## 2. SAE-Based Feature Analysis

The `SAE-Based Feature Analysis` folder contains the scripts used to analyze the activation-level effects of steering using a Sparse Autoencoder (SAE).

* **`extracting.py`**
  Extracts **Layer 0 activations corresponding to answer tokens**. These activations are used as the training data for the SAE.

* **`training.py`**
  Trains the **Sparse Autoencoder (SAE)** using the activations extracted in the previous step.

* **`testing.py`**
  Uses an example pair of **baseline and steered responses** to analyze their encoded activations and identify the most significant SAE features.

The scripts broadly follow the order:

extracting.py
      ↓
training.py
      ↓
testing.py


