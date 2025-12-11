# ✨ Reconnaissance de chiffres manuscrits (Deep Learning)  

[![Licence](https://img.shields.io/badge/license-MIT-blue)](LICENSE)  
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-orange)](http://yann.lecun.com/exdb/mnist/)  

**Nom :** Diangou Camara  
**Projet :** Deep Learning — IIM Digital School  
**Démo en ligne :** [https://diangou.github.io/deep-learning-iim/](https://diangou.github.io/deep-learning-iim/)

---

## 🚀 Contexte

Ce projet utilise les **réseaux de neurones convolutionnels (CNN)** pour reconnaître les chiffres manuscrits du dataset **MNIST**.  

Objectifs :  
- Construire, entraîner et évaluer un modèle CNN pour la classification d’images  
- Exporter le modèle en **ONNX**  
- Créer une application web interactive pour tester les prédictions en temps réel  

> ℹ️ Pour exécuter l’entraînement et tester le modèle, il est recommandé d’ouvrir le notebook sur **[Google Colab](https://colab.research.google.com/)**.

---

## 🧩 Fonctionnalités

- Entraînement d’un modèle CNN en PyTorch  
- Normalisation des données (mean & std)  
- Export du modèle au format **ONNX**  
- Application web interactive :  
  - Dessiner des chiffres  
  - Prédire le chiffre dessiné  
  - Afficher la probabilité par classe  
  - Valider les prédictions correctes avec un compteur  

---

## 📦 Structure du projet

deep-learning-iim/
├── app.js # Script web pour prédiction
├── DeepLearning_Diangou_CAMARA.ipynb # Fichier google collab
├── index.html # Interface graphique
├── styles.css # Styles CSS
└── README.md # Description du projet

---

## 🖥️ Technologies

- Python / PyTorch  
- ONNX & ONNX Runtime Web  
- JavaScript, HTML, CSS  
- MNIST dataset  
