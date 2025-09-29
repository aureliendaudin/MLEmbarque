# ML Model Transpiler to C

Un transpileur universel qui convertit des modèles de machine learning entraînés (sklearn) en code C optimisé, permettant le déploiement sur des systèmes embarqués ou des environnements sans Python.

## Fonctionnalités

- **Transpilation automatique** de modèles ML vers code C natif
- **Support multi-modèles** : régression linéaire, régression logistique, arbres de décision
- **Interface en ligne de commande** simple et intuitive
- **Code C optimisé** avec fonctions mathématiques intégrées
- **Compilation automatique** optionnelle
- **Aucune dépendance sklearn** dans le transpileur final

## Prérequis

### Pour l'entraînement des modèles
```bash
pip install scikit-learn pandas numpy joblib
```

### Pour la transpilation
```bash
pip install joblib numpy
```

### Pour la compilation
```bash
# Linux/Mac
sudo apt-get install gcc  # Ubuntu/Debian
brew install gcc          # macOS
```

## Types de modèles supportés

| Type de modèle | Classe sklearn | Code C généré |
|----------------|----------------|---------------|
| `linear_regression` | `LinearRegression` | `linear_regression_prediction()` |
| `logistic_regression` | `LogisticRegression` | `logistic_regression()` + `sigmoid()` |
| `decision_tree` | `DecisionTreeClassifier/Regressor` | Arbre de décision optimisé |

## Utilisation

### Interface en ligne de commande

```bash
python transpiler_simple_model.py <model_path> <model_type> [options]
```

#### Arguments obligatoires
- `model_path` : Chemin vers le fichier `.joblib` du modèle
- `model_type` : Type de modèle (`linear_regression`, `logistic_regression`, `decision_tree`)

#### Options
- `-o, --output` : Fichier de sortie .c (défaut: `[model_name]_model.c`)
- `--test-data` : Données de test personnalisées (format: `1.0,2.0,3.0`)
- `--compile` : Compiler automatiquement le code généré

### Exemples d'utilisation

#### 1. Régression linéaire simple
```bash
# Transpiler seulement
python transpiler_simple_model.py linear_regression.joblib linear_regression

# Transpiler avec compilation automatique
python transpiler_simple_model.py linear_regression.joblib linear_regression --compile

# Avec données de test personnalisées
python transpiler_simple_model.py linear_regression.joblib linear_regression --test-data 120.5,3,1
```

#### 2. Régression logistique
```bash
python transpiler_simple_model.py logistic_model.joblib logistic_regression -o classifier.c --compile
```

#### 3. Arbre de décision
```bash
python transpiler_simple_model.py tree_model.joblib decision_tree --test-data 1.5,-0.5,2.1 --compile
```

## Workflow complet

### 1. Entraîner un modèle

```python
# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Charger les données
df = pd.read_csv('data.csv')
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# Sauvegarder
joblib.dump(model, "my_model.joblib")
```

### 2. Transpiler vers C

```bash
python transpiler_simple_model.py my_model.joblib linear_regression --compile
```

### 3. Utiliser le code C

```bash
# Le fichier my_model_model.c est généré
# Compilation automatique si --compile était utilisé
./my_model_model
```


## Compilation manuelle

Si la compilation automatique échoue :

```bash
# Compilation standard
gcc -o model prediction_model.c -lm

# Avec optimisations
gcc -O3 -o model prediction_model.c -lm

# Pour systèmes embarqués (exemple ARM)
arm-linux-gnueabi-gcc -static -o model prediction_model.c -lm
```

## Fonctions C implémentées

### Régression linéaire
```c
float linear_regression_prediction(float* features, float* thetas, int n_parameters);
```

### Régression logistique
```c
float logistic_regression(float* features, float* thetas, int n_parameter);
float sigmoid(float x);
float exp_approx(float x, int n_term);  // Approximation de exp() par série de Taylor
```

### Arbres de décision
Code généré dynamiquement selon la structure de l'arbre avec des conditions `if/else` optimisées.
