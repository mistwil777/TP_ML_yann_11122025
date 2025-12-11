# TP Machine Learning - Apprentissage Supervisé & ML Responsable

**Auteur :** Wilfried LEROULIER [Architecte SMA (solutions multi agentique) Capgemini] 
**Date :** 11 Décembre 2025  
**Contexte :** Travaux Pratiques de Machine Learning (Master/École d'Ingénieurs)

---

## 📋 Vue d'ensemble

Ce projet regroupe deux parties complémentaires d'un TP de Machine Learning :
- **Partie 1 :** Apprentissage supervisé avec arbres de décision (détection d'intrusion réseau)
- **Partie 2 :** Machine Learning Responsable & Audit de Fairness (scoring de crédit)

---

## 🗂️ Structure du Projet

```
TP_ML_yann_11122025/
│
├── data/
│   ├── Portmap.csv                    # Dataset trafic réseau (Partie 1)
│   └── synthetic_credit_scoring.csv   # Dataset crédit synthétique (Partie 2)
│
├── TP_Supervise_Arbre.ipynb           # Notebook Partie 1 - Arbres de décision
├── TP_Partie2_Fairness.ipynb          # Notebook Partie 2 - Fairness & SHAP
├── generate_dataset.py                # Script de génération de données (Partie 2)
├── requirements.txt                   # Dépendances Python
└── README.md                          # Ce fichier
```

---

## 🎯 Partie 1 : Apprentissage Supervisé - Détection d'Intrusion

### Objectif
Utiliser un **arbre de décision** pour classifier du trafic réseau et détecter des attaques de type Portmap.

### Dataset
- **Source :** `data/Portmap.csv` (trafic réseau capturé)
- **Features :** Flow Duration, Packets, Bytes, Ports, Flags, etc.
- **Target :** Label (BENIGN vs Attaque)

### Démarche

1. **Chargement & Nettoyage**
   - Analyse des types de données
   - Suppression des colonnes non-numériques (IPs, Timestamps)
   - Encodage du label avec `LabelEncoder`
   - Gestion des valeurs manquantes et infinies
   - Standardisation avec `StandardScaler`

2. **Modélisation**
   - Division Train/Test (80/20)
   - Entraînement d'un `DecisionTreeClassifier`
   - Limitation de la profondeur (`max_depth=4`) pour la lisibilité

3. **Évaluation**
   - Matrice de confusion
   - Classification report (Précision, Recall, F1-Score)
   - **Visualisation de l'arbre** : graphique permettant de comprendre les règles de décision
   - Importance des features

### Concepts Clés
- ✅ **Préparation des données** : Nettoyage, encodage, standardisation obligatoires
- ✅ **Arbre de décision** : Modèle interprétable, visualisable, sensible à l'overfitting
- ✅ **Avantages vs KNN** : Pas besoin de standardisation, rapide en prédiction, facile à expliquer
- ✅ **Ensemble Learning** : Agrégation de modèles (Random Forest) pour réduire la variance

### Résultats
Le modèle obtient une accuracy élevée et identifie correctement les attaques Portmap grâce à des règles simples sur la durée des flux et le nombre de paquets.

---

## ⚖️ Partie 2 : Machine Learning Responsable & Fairness

### Objectif
Auditer un modèle de Machine Learning pour **détecter et mesurer des biais discriminatoires** dans un contexte de scoring de crédit.

### Problème Initial : Absence de Dataset

**Défi :** Le dataset était non disponible.

**Solution Adoptée : Génération de Données Synthétiques**

Nous avons créé un script Python (`generate_dataset.py`) qui :
- Génère 2000 lignes de données de crédit réalistes avec `Faker`
- Introduit **volontairement un biais mathématique** contre les femmes (-15 points de score)
- Simule une discrimination systémique observable dans les données historiques
- Permet la **reproductibilité** et l'**auditabilité** du TP

**Pourquoi cette approche ?**
- ✅ Contrôle total sur le type et l'ampleur du biais
- ✅ Transparence sur l'origine des données
- ✅ Respect de la vie privée (pas de vraies données personnelles)
- ✅ Pédagogie : on sait exactement ce que le modèle devrait détecter

### Dataset Synthétique
- **Features :** Age, Gender, Income, Debt_Ratio, Employment_Years, Loan_Amount
- **Target :** Loan_Approved (0 = Refusé, 1 = Approuvé)
- **Biais introduit :** Les femmes ont ~12-15% de chances en moins d'obtenir un prêt à mérite égal

### Démarche

1. **Génération & Transparence**
   - Exécution de `generate_dataset.py`
   - Documentation de la méthode de génération dans le notebook
   - Visualisation du biais brut dans les données (graphique par genre)

2. **Modélisation**
   - Entraînement d'un `RandomForestClassifier`
   - Le modèle apprend sur des données biaisées (situation réaliste)

3. **Audit de Fairness**
   - **Recall par groupe** : Mesure de la sensibilité pour Hommes vs Femmes
   - **Disparate Impact** : Ratio d'approbation (Femmes / Hommes)
   - **Règle des 80%** : Si DI < 0.8 → Discrimination significative

4. **Explicabilité (SHAP)**
   - Calcul des valeurs SHAP avec `TreeExplainer`
   - Summary plot : Identification des features les plus influentes
   - Vérification si le genre est directement utilisé pour discriminer

### Concepts Clés

#### 🔍 Audit de Fairness
- **Disparate Impact** : Métrique légale (norme USA) mesurant les discriminations
- **Group Fairness** : Comparaison des performances entre groupes protégés
- **Recall par groupe** : Évite qu'une bonne accuracy globale cache des disparités

#### 🧠 Explicabilité (SHAP)
- **SHAP Values** : Mesure de la contribution de chaque feature à chaque prédiction
- **Discrimination directe** : Le modèle utilise explicitement le genre
- **Proxy discrimination** : Le modèle utilise des variables corrélées au genre (salaire, emploi)

#### ⚠️ Leçons Éthiques
1. **Un algorithme n'est jamais neutre** : Il reflète les biais des données
2. **L'accuracy ne suffit pas** : Un modèle précis peut être injuste
3. **L'audit est obligatoire** : Sans mesures de fairness, les biais passent inaperçus
4. **La transparence protège** : Documenter les choix et limitations est essentiel

### Résultats
- Le modèle atteint une bonne précision globale (~85%)
- **Mais** le Disparate Impact est < 0.8 → Discrimination détectée
- SHAP révèle que le genre est une variable influente dans les décisions
- **Conclusion** : Le modèle a appris et reproduit le biais discriminatoire

---

## 🛠️ Installation & Utilisation

### Prérequis
- Python 3.8+
- VS Code (recommandé)
- Terminal Bash (Git Bash sur Windows)

### Installation

```bash
# 1. Créer l'environnement virtuel
python -m venv venv

# 2. Activer l'environnement (Windows Bash)
source venv/Scripts/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Exécution

#### Partie 1 - Arbres de Décision
```bash
# Ouvrir le notebook
jupyter notebook TP_Supervise_Arbre.ipynb

# Exécuter les cellules séquentiellement
```

#### Partie 2 - Fairness
```bash
# 1. Générer le dataset synthétique
python generate_dataset.py

# 2. Ouvrir le notebook
jupyter notebook TP_Partie2_Fairness.ipynb

# 3. Exécuter les cellules séquentiellement
```

---

## 📦 Dépendances Principales

| Package | Usage |
|---------|-------|
| `pandas` | Manipulation de données |
| `numpy` | Calculs numériques |
| `scikit-learn` | Modèles ML, métriques |
| `matplotlib` | Visualisations de base |
| `seaborn` | Visualisations avancées |
| `shap` | Explicabilité des modèles |
| `fairlearn` | Audit de fairness (Partie 2) |
| `faker` | Génération de fausses données |
| `jupyter` | Notebooks interactifs |

---

## 🎓 Concepts Fondamentaux à Retenir

### Machine Learning Supervisé

1. **Préparation des données = 80% du travail**
   - Nettoyage, encodage, gestion des manquants
   - Standardisation pour la plupart des algos (pas pour les arbres)
   - Pas de données → Pas de modèle fonctionnel

2. **Trade-off Interprétabilité vs Performance**
   - Arbres : Faciles à expliquer, visualisables
   - Random Forest : Plus performantes, moins lisibles
   - Réseaux de neurones : Très performants, boîtes noires

3. **Validation rigoureuse**
   - Train/Test split obligatoire
   - Matrice de confusion pour comprendre les erreurs
   - Métriques multiples (Accuracy, Precision, Recall, F1)

### Machine Learning Responsable

4. **Biais algorithmiques**
   - Les algorithmes reflètent les biais des données historiques
   - L'équité ne vient pas "par défaut"
   - L'audit est une obligation éthique et légale

5. **Métriques de Fairness**
   - **Disparate Impact** : Ratio de succès entre groupes
   - **Equal Opportunity** : Recall équilibré entre groupes
   - **Calibration** : Confiance équitable dans les prédictions

6. **Explicabilité**
   - SHAP : Valeurs de Shapley pour l'attribution de prédiction
   - LIME : Explication locale par approximation linéaire
   - Importance des features : Quelles variables comptent le plus ?

7. **Gouvernance & Documentation**
   - Documenter les choix de design
   - Tracer l'origine des données
   - Rendre le système auditable
   - Prévoir une supervision humaine pour les décisions critiques

---

## 🔄 Démarche Scientifique Appliquée

### Problème → Solution

| Problème Rencontré | Solution Adoptée |
|-------------------|-----------------|
| Dataset Portmap incomplet (colonnes manquantes) | Détection automatique des colonnes disponibles |
| Colonnes non-numériques (IPs) | Suppression automatique via `dtype == 'object'` |
| Valeurs infinies dans le dataset | Remplacement par NaN puis imputation par la moyenne |
| Dataset de fairness inexistant | Génération synthétique contrôlée avec biais intentionnel |
| Version de SHAP incertaine | Détection automatique de la structure (list vs array) |

### Bonnes Pratiques Appliquées
- ✅ Code commenté et cellules Markdown explicatives
- ✅ Gestion d'erreurs robuste (try/except implicites)
- ✅ Visualisations claires avec légendes et titres
- ✅ Transparence sur les limitations et choix méthodologiques
- ✅ Reproductibilité (seeds, scripts de génération)

---

## 📚 Références & Ressources

### Documentation Technique
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Fairlearn Guide](https://fairlearn.org/)

### Articles Académiques
- "Fairness and Machine Learning" - Barocas, Hardt, Narayanan (2019)
- "A Survey on Bias and Fairness in Machine Learning" - Mehrabi et al. (2021)

### Outils de Gouvernance
- [AI Ethics Guidelines](https://ec.europa.eu/digital-strategy/en/policies/expert-group-ai)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

---

## 🎯 Compétences Développées

### Techniques
- ✅ Manipulation de données avec Pandas
- ✅ Modélisation ML avec Scikit-Learn
- ✅ Visualisation avec Matplotlib/Seaborn
- ✅ Explicabilité avec SHAP
- ✅ Audit de fairness algorithmique
- ✅ Génération de données synthétiques

### Méthodologiques
- ✅ Démarche scientifique (Problème → Hypothèse → Expérimentation → Conclusion)
- ✅ Documentation technique
- ✅ Gestion de versions (Git)
- ✅ Reproductibilité des expériences
- ✅ Pensée critique sur les biais

### Éthiques & Professionnelles
- ✅ Conscience des enjeux de discrimination algorithmique
- ✅ Transparence sur les limitations des modèles
- ✅ Respect de la vie privée (données synthétiques)
- ✅ Responsabilité dans le déploiement de systèmes ML

---

## ⚠️ Limitations & Perspectives

### Limitations Actuelles
- Dataset synthétique simplifié (biais unidimensionnel)
- Pas de correction du biais appliquée (seulement détection)
- Pas de test sur données réelles (RGPD)
- Modèles simples (pas de deep learning)

### Pistes d'Amélioration
- Implémenter des techniques de debiasing (Fairlearn, reweighting)
- Tester d'autres modèles (XGBoost, réseaux de neurones)
- Ajouter une interface de démonstration (Streamlit)
- Comparer plusieurs métriques de fairness (Equalized Odds, Calibration)
- Implémenter un monitoring continu du biais en production

---

## 👤 Auteur & Contact

**Wilfried LEROULIER**  
Étudiant en IA  
📧 Contact : mistwil777 
📧 Mail : wilfried.leroulier@ecoles-epsi.net

**Note :** Ce projet est réalisé dans un cadre pédagogique. Les données sont synthétiques et ne reflètent aucune situation réelle. L'objectif est d'apprendre les techniques d'audit de fairness dans un environnement contrôlé.

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique. Le code est mis à disposition à des fins éducatives.

---

## 🙏 Remerciements

- Enseignant du cours de Machine Learning : Yann Causeur
- Communautés open-source (Scikit-Learn, SHAP, Fairlearn)
- Dataset Portmap original (pour la Partie 1)

---

**Dernière mise à jour :** 11 Décembre 2025
