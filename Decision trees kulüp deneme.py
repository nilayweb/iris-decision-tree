# Gerekli Kütüphaneleri Yükleyelim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Veri setini yükleyelim
iris = datasets.load_iris()
x = iris.data # özellikler (sepal, petal uzunlukları)
y = iris.target # hedef değişken (çiçek türleri)

# Eğitim ve test setine bölelim
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

# Decision trees modelini oluşturalım ve eğitelim
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

# Modelin doğruluğunu test edelim
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")

# Decision trees i görselleştirelim
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()