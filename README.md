### Modèle de Perceptron à 3 couches

#### Description
Ce modèle implémente un perceptron à 3 couches avec une couche d'entrée, une couche cachée et une couche de sortie. Il utilise des fonctions d'activation ReLU pour la couche cachée et softmax pour la couche de sortie. Le modèle est entraîné à l'aide d'une descente de gradient stochastique.


Le fichier `main.py` contient la classe model et le fichier `note_book` permet le teste des fonctions separements

#### Fonctionnalités
- Initialisation aléatoire des poids et des biais
- Fonctions d'activation ReLU et softmax
- Calcul de la précision du modèle
- Entraînement du modèle avec des données d'entraînement
- Sauvegarde et chargement du modèle à partir d'un fichier

#### Utilisation
**Installation**
Assurez-vous d'avoir Python installé sur votre système.

**Entraînement du modèle**
1. Créez une instance du modèle en spécifiant les tailles des couches d'entrée, cachée et de sortie.
2. Appelez la méthode `fit` avec les données d'entraînement, le nombre d'itérations, le taux d'apprentissage et d'autres paramètres facultatifs.
3. Le dataset qui de base est fait 42000 lignes a ete tronquer pour le l'occation et il fait donc 1000 lignes pour le test et 200 pour le training
Exemple :
```python
from three_layer_perceptron import Model

# Création du modèle
model = Model(input_size=784, hidden_size=100, output_size=10)

# Chargement des données d'entraînement
x_train = ...
y_train = ...

# Entraînement du modèle
model.fit(x_train, y_train, eval=10, iters=100, a=0.1, show_training_info=True)
```

**Utilisation du modèle entraîné**


**N'oublier pas de Mettre a Jour les chemins d'acces selon vos besoins**

Utilisez la méthode `predict` pour obtenir les prédictions du modèle pour de nouvelles données.

Exemple :
```python
# Prédictions pour de nouvelles données
x_new = ...
predictions = model.predict(x_new)
```

**Sauvegarde et chargement du modèle**
Utilisez les méthodes `save` et `load` pour sauvegarder et charger le modèle depuis un fichier.

Exemple :
```python
# Sauvegarde du modèle
model.save('model.pkl')

# Chargement du modèle sauvegardé
loaded_model = Model.load('model.pkl')
```
Un fichier `model.pkl` est fournie dans le repertoire pour eviter de perdre du temps en entrainement,  vous pouvez juste charger ce model et faire des predictions

Adapter les chemins d'accès et les exemples de code en fonction de votre environnement et de vos données.
