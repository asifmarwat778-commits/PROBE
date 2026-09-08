# PROBE

**Multi-Probe Quantification of Radical Speciation in Fe(III)- and Chlorine-Based Advanced Oxidation Processes**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

Codes and datasets for the five-probe method to simultaneously determine the steady-state concentrations of multiple radicals (•OH, Cl•, Cl₂•⁻, ClO•, and Fe(IV)=O²⁺) in HOCl/Fe(III), UV₂₅₄/Fe(III), UV₂₅₄/HOCl, and UV₂₅₄/HOCl/Fe(III) systems, with DFT analysis, Kintecus kinetic validation, and GBR-based machine learning prediction (R² = 0.993, MAE = 0.015).

## Highlights

- Simultaneous quantification of 5 radical species using a 5-probe array (1,4-DMOB, BA, CBZ, NB, PMSO)
- Probe selectivity validated by DFT (condensed Fukui indices, ΔG‡, ΔG) at B3LYP/def2-SVP (SMD, water)
- Kintecus V5.75 kinetic modeling cross-validation
- Machine learning prediction of probe degradation (Ct/C0): GBR, SVR, RFR, HGBR, MLP, KNN
- Leave-one-pH-out / leave-one-ion-out cross-validation for model boundary definition
- Graphical user interface (GUI) for instantaneous, solvent-free Ct/C0 prediction

## Repository Structure
