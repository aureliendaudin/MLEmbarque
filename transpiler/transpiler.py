#!/usr/bin/env python3
"""
Transpileur universel pour modèles ML vers code C
Supporte: régression linéaire, régression logistique, arbres de décision
"""

import joblib
import numpy as np
import argparse
import sys
import os

def generate_tree_code(tree_data, node_id=0, depth=0):
    """
    Génère récursivement le code C pour un arbre de décision
    tree_data: dictionnaire contenant les données de l'arbre
    """
    children_left = tree_data['children_left']
    children_right = tree_data['children_right']
    feature = tree_data['feature']
    threshold = tree_data['threshold']
    value = tree_data['value']
    n_features = tree_data['n_features']

    # Indentation pour la lisibilité
    indent = "    " * (depth + 1)

    # Si c'est une feuille
    if children_left[node_id] == children_right[node_id]:
        if len(value[node_id][0]) == 1:  # Régression
            return f"{indent}return {value[node_id][0][0]:.6f};"
        else:  # Classification - retourner la classe majoritaire
            class_idx = np.argmax(value[node_id][0])
            return f"{indent}return {class_idx};"

    # Nœud de décision
    feature_idx = feature[node_id]
    threshold_val = threshold[node_id]

    code = f"{indent}if (features[{feature_idx}] <= {threshold_val:.6f}) {{\n"
    code += generate_tree_code(tree_data, children_left[node_id], depth + 1) + "\n"
    code += f"{indent}}} else {{\n"
    code += generate_tree_code(tree_data, children_right[node_id], depth + 1) + "\n"
    code += f"{indent}}}"

    return code

def extract_model_data(model, model_type):
    """
    Extrait les données du modèle selon son type
    """
    if model_type == "linear_regression":
        return {
            'coef_': model.coef_,
            'intercept_': model.intercept_,
            'n_features': len(model.coef_)
        }

    elif model_type == "logistic_regression":
        coeff = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_

        return {
            'coef_': coeff,
            'intercept_': intercept,
            'n_features': len(coeff)
        }

    elif model_type == "decision_tree":
        tree = model.tree_
        return {
            'children_left': tree.children_left,
            'children_right': tree.children_right,
            'feature': tree.feature,
            'threshold': tree.threshold,
            'value': tree.value,
            'n_features': tree.n_features_,
            'node_count': tree.node_count,
            'is_classifier': hasattr(model, 'classes_')
        }

    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")

def generate_c_code(model_data, model_type, test_data=None):
    """
    Génère le code C pour le modèle donné
    """
    # En-tête C commun
    c_code = "#include <stdio.h>\n#include <math.h>\n\n"

    # Fonctions utilitaires pour la régression logistique
    if model_type == "logistic_regression":
        c_code += """float exp_approx(float x, int n_term) {
    float result = 0.0f;
    float term = 1.0f;

    for (int i = 0; i <= n_term; i++) {
        result += term;
        if (i < n_term) {
            term = term * x / (i + 1);
        }
    }

    return result;
}

float sigmoid(float x) {
    // sigmoid(x) = 1 / (1 + e^(-x))
    float exp_minus_x = exp_approx(-x, 10);
    return 1.0f / (1.0f + exp_minus_x);
}

float logistic_regression(float* features, float* thetas, int n_parameter) {
    float z = thetas[0];

    for (int i = 1; i < n_parameter; i++) {
        z += thetas[i] * features[i-1];
    }

    return sigmoid(z);
}

"""

    # Fonction de régression linéaire
    if model_type == "linear_regression":
        c_code += """float linear_regression_prediction(float* features, float* thetas, int n_parameters) {
    float prediction = thetas[0];

    for (int i = 1; i < n_parameters; i++) {
        prediction += thetas[i] * features[i-1];
    }

    return prediction;
}

"""

    # Générer la fonction prediction selon le type de modèle
    if model_type == "linear_regression":
        coeff = model_data['coef_']
        intercept = model_data['intercept_']
        n_features = model_data['n_features']

        # Paramètres theta = [intercept, coef1, coef2, ...]
        thetas = [intercept] + list(coeff)
        thetas_str = "{" + ", ".join([f"{theta:.6f}f" for theta in thetas]) + "}"

        c_code += f"float prediction(float *features, int n_features) {{\n"
        c_code += f"    if (n_features != {n_features}) {{\n"
        c_code += f"        printf(\"Erreur: nombre de features incorrect. Attendu: {n_features}, recu: %d\\n\", n_features);\n"
        c_code += f"        return -1.0;\n"
        c_code += f"    }}\n\n"
        c_code += f"    float thetas[{len(thetas)}] = {thetas_str};\n"
        c_code += f"    return linear_regression_prediction(features, thetas, {len(thetas)});\n"
        c_code += f"}}\n\n"

        # Données de test par défaut
        if test_data is None:
            test_data = [1.0] * n_features

    elif model_type == "logistic_regression":
        coeff = model_data['coef_']
        intercept = model_data['intercept_']
        n_features = model_data['n_features']

        # Paramètres theta = [intercept, coef1, coef2, ...]
        thetas = [intercept] + list(coeff)
        thetas_str = "{" + ", ".join([f"{theta:.6f}f" for theta in thetas]) + "}"

        c_code += f"float prediction(float *features, int n_features) {{\n"
        c_code += f"    if (n_features != {n_features}) {{\n"
        c_code += f"        printf(\"Erreur: nombre de features incorrect. Attendu: {n_features}, recu: %d\\n\", n_features);\n"
        c_code += f"        return -1.0;\n"
        c_code += f"    }}\n\n"
        c_code += f"    float thetas[{len(thetas)}] = {thetas_str};\n"
        c_code += f"    return logistic_regression(features, thetas, {len(thetas)});\n"
        c_code += f"}}\n\n"

        # Données de test par défaut
        if test_data is None:
            test_data = [0.0] * n_features

    elif model_type == "decision_tree":
        n_features = model_data['n_features']
        is_classifier = model_data['is_classifier']

        # Générer le code de l'arbre
        tree_code = generate_tree_code(model_data)

        return_type = "int" if is_classifier else "float"

        c_code += f"{return_type} prediction(float *features, int n_features) {{\n"
        c_code += f"    if (n_features != {n_features}) {{\n"
        c_code += f"        printf(\"Erreur: nombre de features incorrect. Attendu: {n_features}, recu: %d\\n\", n_features);\n"
        c_code += f"        return -1;\n"
        c_code += f"    }}\n\n"
        c_code += tree_code + "\n"
        c_code += f"}}\n\n"

        # Données de test par défaut
        if test_data is None:
            test_data = [0.0] * n_features

    # Fonction main
    test_data_str = "{" + ", ".join([f"{x:.6f}f" for x in test_data]) + "}"

    c_code += f"int main() {{\n"
    c_code += f"    float test_features[{len(test_data)}] = {test_data_str};\n\n"
    c_code += f"    printf(\"Test du modele {model_type}\\n\");\n"
    c_code += f"    printf(\"Features: \");\n"
    c_code += f"    for(int i = 0; i < {len(test_data)}; i++) {{\n"
    c_code += f"        printf(\"%.3f \", test_features[i]);\n"
    c_code += f"    }}\n"
    c_code += f"    printf(\"\\n\");\n\n"

    if model_type == "decision_tree" and model_data.get('is_classifier', False):
        c_code += f"    int predicted_class = prediction(test_features, {len(test_data)});\n"
        c_code += f"    printf(\"Classe predite: %d\\n\", predicted_class);\n"
    else:
        c_code += f"    float predicted_value = prediction(test_features, {len(test_data)});\n"
        c_code += f"    printf(\"Valeur predite: %.6f\\n\", predicted_value);\n"

    c_code += f"\n    return 0;\n}}"

    return c_code

def transpile_model(model_path, model_type, output_path=None, test_data=None, compile_model=False):
    """
    Fonction principale de transpilation
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle {model_path} n'existe pas")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base_name}_model.c"

    print(f"Chargement du modèle: {model_path}")
    print(f"Type de modèle: {model_type}")

    # Charger le modèle
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")

    # Extraire les données du modèle
    try:
        model_data = extract_model_data(model, model_type)
        print(f"Modèle chargé avec {model_data.get('n_features', 'N/A')} features")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'extraction des données: {e}")

    # Générer le code C
    try:
        c_code = generate_c_code(model_data, model_type, test_data)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération du code: {e}")

    # Sauvegarder le code C
    try:
        with open(output_path, 'w') as f:
            f.write(c_code)
        print(f"Code C généré: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la sauvegarde: {e}")

    # Afficher les commandes
    exec_name = os.path.splitext(output_path)[0]
    compile_command = f"gcc -o {exec_name} {output_path} -lm"

    print(f"\nCommande de compilation:")
    print(f"  {compile_command}")
    print(f"Commande d'exécution:")
    print(f"  ./{exec_name}")

    # Compilation automatique si demandée
    if compile_model:
        import subprocess
        try:
            result = subprocess.run(compile_command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Compilation réussie: {exec_name}")

                # Exécution automatique
                exec_result = subprocess.run([f"./{exec_name}"], capture_output=True, text=True)
                if exec_result.returncode == 0:
                    print(f"\nSortie du programme:")
                    print(exec_result.stdout)
                else:
                    print(f"Erreur d'exécution: {exec_result.stderr}")
            else:
                print(f"Erreur de compilation: {result.stderr}")
        except FileNotFoundError:
            print("gcc non trouvé, compilation manuelle requise")

    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Transpileur de modèles ML vers code C",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python transpiler.py model.joblib linear_regression
  python transpiler.py model.joblib logistic_regression -o output.c
  python transpiler.py tree.joblib decision_tree --compile
  python transpiler.py model.joblib linear_regression --test-data 1.5,2.0,3.0
        """
    )

    parser.add_argument("model_path", help="Chemin vers le fichier .joblib du modèle")
    parser.add_argument("model_type", 
                       choices=["linear_regression", "logistic_regression", "decision_tree"],
                       help="Type de modèle")
    parser.add_argument("-o", "--output", 
                       help="Fichier de sortie .c (défaut: [model_name]_model.c)")
    parser.add_argument("--test-data", 
                       help="Données de test séparées par des virgules (ex: 1.0,2.0,3.0)")
    parser.add_argument("--compile", action="store_true",
                       help="Compiler automatiquement le code généré")

    args = parser.parse_args()

    # Parser les données de test
    test_data = None
    if args.test_data:
        try:
            test_data = [float(x.strip()) for x in args.test_data.split(',')]
        except ValueError:
            print("Erreur: format des données de test invalide")
            sys.exit(1)

    try:
        transpile_model(
            model_path=args.model_path,
            model_type=args.model_type,
            output_path=args.output,
            test_data=test_data,
            compile_model=args.compile
        )
        print("\n✓ Transpilation terminée avec succès!")

    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
