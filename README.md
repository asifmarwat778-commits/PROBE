<img width="1280" height="720" alt="Image" src="https://github.com/user-attachments/assets/cfc31ee9-c5df-450c-a565-abc1ba2c7276" />

## PROBE

PROBE (Predictive Radical quantification via Oxidant Benchmarking and Estimation) Multi-Probe Quantification of Radical Speciation in Fe(III)- and Chlorine-Based Advanced Oxidation Processes
## Description
Codes and datasets for the five-probe method to simultaneously determine the steady-state concentrations of multiple radicals (•OH, Cl•, Cl₂•⁻, ClO•, and Fe(IV)=O²⁺) in HOCl/Fe(III), UV₂₅₄/Fe(III), UV₂₅₄/HOCl, and UV₂₅₄/HOCl/Fe(III) systems, with DFT analysis, Kintecus kinetic validation, and GBR-based machine learning prediction (R² = 0.993, MAE = 0.015).

## Using PROBE

The tool is available in the script app.py. It is intended to be run locally via Python. The user can select the target probe (1,4-DMOB, BA, CBZ, NB, or PMSO) and AOP system (HOCl/Fe(III), UV254/Fe(III), UV254/HOCl, or UV254/HOCl/Fe(III)), enter the operational parameters (reaction time, solution pH, HOCl and Fe(III) dose, UV intensity, and water-matrix ion concentrations), and the GUI will output the predicted probe degradation (Ct/C0) in real time based on the trained Gradient Boosting Regressor (GBR, R2 = 0.993), providing a rapid, solvent-free alternative to conventional chromatographic quantification within the validated pH 3-9 domain.

## Training Dataset
The complete datasets of experimentally measured kobs values used to train FADE are given in the files `Data2026042605 ONP HOCl.csv` (391 data points) and `Data_NH2Cl.csv` (863 data points). The GUI code is given in the file attached here named FADE_GUI_XGBR_UV_NH2Cl_Fe(III).ipynb and 



## Highlights

•	Five-probe array quantifies •OH, Cl•, Cl2•⁻, ClO•, Fe(IV) in Fe(III)-chlorine mediated AOPs.
•	DFT-validated probe selectivity via Fukui indices and activation energies.
•	Kintecus modeling validates experimental steady-state radical concentrations.
•	Ternary UV254/HOCl/Fe(III) yields diverse radical speciation peaking at pH 3.
•	GBR machine learning predicts probe degradation (R2 = 0.993) via an interactive GUI.

## Repository Structure
## MIT License

Copyright (c) 2026 Muhammad Asif, Aiwen Wang, Wei Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
