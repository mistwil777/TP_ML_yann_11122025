"""
Script de génération de dataset synthétique pour l'audit de biais ML
====================================================================

Ce script génère un dataset de "Credit Scoring" avec un biais volontaire
contre les femmes, afin de démontrer les capacités d'audit des outils
de fairness en Machine Learning.

Usage:
    python generate_dataset.py

Output:
    data/synthetic_credit_scoring.csv (2000 lignes)
"""

import pandas as pd
import numpy as np
from faker import Faker
import random

# Configuration
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

N_SAMPLES = 2000

print("=" * 60)
print("GÉNÉRATION DE DATASET SYNTHÉTIQUE - CREDIT SCORING")
print("=" * 60)

# Génération des données de base
data = {
    'ID': [f'CLIENT_{i:05d}' for i in range(1, N_SAMPLES + 1)],
    'Age': np.random.randint(22, 70, N_SAMPLES),
    'Gender': np.random.choice(['Male', 'Female'], N_SAMPLES, p=[0.52, 0.48]),
    'Income': np.random.normal(45000, 20000, N_SAMPLES).clip(18000, 150000),
    'Debt_Ratio': np.random.uniform(0.05, 0.85, N_SAMPLES),
    'Employment_Years': np.random.randint(0, 35, N_SAMPLES),
    'Loan_Amount': np.random.normal(25000, 15000, N_SAMPLES).clip(5000, 100000)
}

df = pd.DataFrame(data)

# Calcul du score de crédit (logique métier)
print("\n[1/4] Calcul des scores de crédit...")

def calculate_credit_score(row):
    """
    Calcule un score de crédit basé sur les caractéristiques financières
    """
    score = 0
    
    # Facteurs positifs
    score += row['Income'] * 0.0005  # Revenu (max ~75 points)
    score += row['Employment_Years'] * 1.5  # Ancienneté (max ~52 points)
    score += (row['Age'] - 22) * 0.5  # Âge/Maturité (max ~24 points)
    
    # Facteurs négatifs
    score -= row['Debt_Ratio'] * 60  # Taux d'endettement (max -51 points)
    score -= row['Loan_Amount'] * 0.0003  # Montant demandé (max -30 points)
    
    return score

df['Credit_Score'] = df.apply(calculate_credit_score, axis=1)

# INTRODUCTION DU BIAIS DISCRIMINATOIRE
print("[2/4] Introduction du biais contre les femmes...")
print("      ⚠️  BIAIS VOLONTAIRE : Malus de -15 points pour Gender='Female'")

# Malus pour les femmes (biais intentionnel)
GENDER_BIAS_MALUS = -15
df.loc[df['Gender'] == 'Female', 'Credit_Score'] += GENDER_BIAS_MALUS

# Conversion du score en probabilité d'approbation
print("[3/4] Génération des décisions d'approbation...")

def score_to_probability(score):
    """
    Convertit le score en probabilité via une fonction sigmoïde
    """
    # Normalisation et sigmoïde
    normalized = (score - 30) / 20  # Centrage
    prob = 1 / (1 + np.exp(-normalized))
    return prob

df['Approval_Probability'] = df['Credit_Score'].apply(score_to_probability)

# Ajout de bruit aléatoire pour le réalisme
noise = np.random.normal(0, 0.05, N_SAMPLES)
df['Approval_Probability'] = (df['Approval_Probability'] + noise).clip(0, 1)

# Décision finale (0 ou 1)
df['Loan_Approved'] = (df['Approval_Probability'] > 0.5).astype(int)

# Statistiques de contrôle
print("\n[4/4] Vérification du biais introduit...")
print("\n" + "=" * 60)
print("STATISTIQUES GLOBALES")
print("=" * 60)

total_approved = df['Loan_Approved'].sum()
print(f"Total de prêts approuvés : {total_approved}/{N_SAMPLES} ({total_approved/N_SAMPLES*100:.1f}%)")

# Statistiques par genre
male_approved = df[df['Gender'] == 'Male']['Loan_Approved'].mean()
female_approved = df[df['Gender'] == 'Female']['Loan_Approved'].mean()

print(f"\nTaux d'approbation Hommes   : {male_approved*100:.1f}%")
print(f"Taux d'approbation Femmes   : {female_approved*100:.1f}%")
print(f"Écart (preuve du biais)     : {(male_approved - female_approved)*100:.1f} points")

disparate_impact = female_approved / male_approved if male_approved > 0 else 0
print(f"Disparate Impact            : {disparate_impact:.3f}")
print(f"                              (valeur idéale = 1.0, ici < 0.8 = biais détectable)")

# Sauvegarde
print("\n" + "=" * 60)
print("SAUVEGARDE DU DATASET")
print("=" * 60)

# Colonnes finales (on supprime les colonnes intermédiaires)
final_columns = ['ID', 'Age', 'Gender', 'Income', 'Debt_Ratio', 
                 'Employment_Years', 'Loan_Amount', 'Loan_Approved']
df_final = df[final_columns]

# Arrondissage pour le réalisme
df_final['Income'] = df_final['Income'].round(2)
df_final['Debt_Ratio'] = df_final['Debt_Ratio'].round(3)
df_final['Loan_Amount'] = df_final['Loan_Amount'].round(2)

# Sauvegarde
output_path = 'data/synthetic_credit_scoring.csv'
df_final.to_csv(output_path, index=False)

print(f"✓ Dataset sauvegardé : {output_path}")
print(f"✓ Dimensions : {df_final.shape[0]} lignes × {df_final.shape[1]} colonnes")
print(f"✓ Biais confirmé : {(male_approved - female_approved)*100:.1f}% d'écart\n")

print("=" * 60)
print("PRÊT POUR L'ANALYSE DANS LE NOTEBOOK !")
print("=" * 60)
