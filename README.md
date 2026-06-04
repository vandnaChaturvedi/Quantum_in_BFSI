# Quantum Finance for BFSI

## Portfolio Optimization and Quantum Risk Analytics

This repository contains a collection of Jupyter notebooks  implementations. Qiskit finance Notebooks : https://github.com/vandnaChaturvedi/qiskit_finance_tutorial
---

## Repository Structure

| File          | Title                                                         
| ------------- | -------------------------------------------------------------- 
| **1_1_Introductory_Quantum_Enhanced_Markowitz.ipynb** | Introductory Quantum-Enhanced Markowitz Portfolio Optimization 
| **2_1_Industry_Style_Quantum_Portfolio_Optimization.ipynb** | Industry-Style Quantum Portfolio Optimization                  
| **1_2_Monte_Carlo_in_Financial_Risk.ipynb** | Monte Carlo Simulation in Financial Risk Management            
| **2_2_Quantum_Amplitude_Estimation_for_Finance.ipynb** | Quantum Amplitude Estimation (QAE) for Financial Applications 

---

## Notebook Overview

### 1_1_Introductory_Quantum_Enhanced_Markowitz.ipynb.ipynb

### Introductory Quantum-Enhanced Markowitz Portfolio Optimization

This notebook introduces the transition from classical portfolio optimization to quantum optimization.

Topics Covered:

* Modern Portfolio Theory (MPT)
* Markowitz Mean-Variance Optimization
* Expected Returns and Covariance Matrix
* Classical Optimization using SciPy
* QUBO Transformation
* MinimumEigenOptimizer
* Quantum Optimization Workflow
* Efficient Frontier Visualization

Learning Outcome:

Understand how classical portfolio optimization problems can be reformulated for quantum algorithms.

---

### 2_1_Industry_Style_Quantum_Portfolio_Optimization.ipynb.ipynb

### Industry-Style Quantum Portfolio Optimization

This notebook demonstrates portfolio optimization workflows inspired by publicly available research from:

* JPMorgan Chase
* IBM Quantum
* Goldman Sachs
* QC Ware
* D-Wave
* BBVA
* Multiverse Computing

Topics Covered:

* Real Market Data Acquisition
* Portfolio Construction
* Cardinality Constraints
* QUBO Formulation
* QAOA
* SamplingVQE
* Hybrid Quantum-Classical Optimization
* Portfolio Performance Evaluation

Metrics:

* Expected Return
* Portfolio Variance
* Sharpe Ratio

Learning Outcome:

Understand how financial institutions explore quantum optimization for large-scale portfolio construction.

---

### 1_2_Monte_Carlo_in_Financial_Risk.ipynb

### Monte Carlo Simulation in Financial Risk Management

This notebook provides an introduction to Monte Carlo methods used in modern financial institutions.

Topics Covered:

* Geometric Brownian Motion
* Stock Price Simulation
* European Option Pricing
* Value-at-Risk (VaR)
* Expected Shortfall (ES)
* Credit Exposure Simulation
* Risk Distribution Analysis

Visualizations:

* Simulated Price Paths
* Loss Distribution
* VaR Thresholds

Learning Outcome:

Understand how Monte Carlo methods are applied to risk analytics and derivatives pricing.

---

### 2_2_Quantum_Amplitude_Estimation_for_Finance.ipynb

### Quantum Amplitude Estimation (QAE) for Financial Applications

This notebook introduces one of the most important quantum algorithms in finance.

Topics Covered:

* Classical Monte Carlo Review
* Quantum Amplitude Estimation Theory
* Amplitude Encoding
* Grover Operator
* Financial Probability Estimation
* Expected Payoff Computation
* Error Analysis

Comparison:

Classical Monte Carlo:

Error ∝ O(1/√N)

Quantum Amplitude Estimation:

Error ∝ O(1/N)

Learning Outcome:

Understand how QAE can potentially provide quadratic speedups for financial risk calculations.

---
## Technology Stack

### Quantum Computing

* Qiskit
* Qiskit Finance
* Qiskit Optimization
* Qiskit Algorithms
* Qiskit Aer

### Classical Finance

* NumPy
* SciPy
* Pandas
* Matplotlib
* yFinance

---

## Installation

```bash
pip install qiskit
pip install qiskit-finance
pip install qiskit-optimization
pip install qiskit-algorithms
pip install qiskit-aer
pip install yfinance
pip install numpy pandas scipy matplotlib
```

---


