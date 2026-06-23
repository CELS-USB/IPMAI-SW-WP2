# IPMAI Cycle Monitor

IPMAI Cycle Monitor is an open-source software suite developed within the **IPMAI (Intelligent Performance Monitoring and Analytics Interface)** project. The software provides a unified environment for simulation, monitoring, feature extraction, machine learning, and explainable analysis of cyclic and dynamical systems.

The suite consists of three independent modules that can be launched individually or through a common graphical launcher.

## Modules

### PID Cycle Monitor

The PID Cycle Monitor simulates closed-loop control systems and enables analysis of monitoring methods under controlled operating conditions.

Users can define and simulate various operating regimes and disturbance scenarios. Available perturbations include high-frequency ripple disturbances, low-order oscillations, modulation effects, impacts, and resonance-like phenomena. Although some perturbation frequencies may exceed the control-loop sampling frequency, their influence is reflected in the measured control-loop variables and can be analysed using the provided monitoring tools.

The module supports:

* PID control-loop simulation
* Generation of synthetic operating scenarios
* Feature extraction from measured variables
* Linear Neural Unit (LNU) modelling
* Learning Entropy (LE) evaluation
* Saliency-based explainability analysis
* Machine-learning-based condition classification

### Robotic Cycle Monitor

The Robotic Cycle Monitor analyses cyclic industrial robot data acquired during repetitive robotic operations.

The module provides:

* Data import and preprocessing
* Cycle-based feature extraction
* LNU-based modelling
* Learning Entropy evaluation
* Weight-space analysis
* Saliency-map generation
* PCA visualization
* MLP-based classification of operating conditions

### Harmonic Drive Cycle Monitor

The Harmonic Drive Cycle Monitor analyses cyclic measurements acquired from harmonic-drive-based robotic systems and related test platforms.

The module supports:

* Time-series analysis
* Cycle-based feature extraction
* LNU modelling
* Learning Entropy evaluation
* Weight-space monitoring
* Saliency-map generation
* PCA analysis
* MLP-based condition classification

## Explainable Machine Learning

Instead of relying solely on measured signals, the software also evaluates the behaviour of learned model parameters and their evolution. The extracted parameter-space representations can be analysed using Learning Entropy, saliency maps, dimensionality-reduction methods, and classification techniques.

The software incorporates:

* Linear Neural Units (LNU)
* Learning Entropy (LE)
* Principal Component Analysis (PCA)
* Saliency-based explainability methods
* Multilayer Perceptron (MLP) classifiers

## Documentation

Each module contains its own detailed documentation generated using Sphinx.

Documentation covers:

* Installation
* Quick start guides
* Data structures
* Feature extraction methods
* LNU modelling
* Learning Entropy
* Saliency maps
* PCA analysis
* Classification workflows
* Software architecture

## Project Background

The software has been developed within the **IPMAI project** under the Interreg Austria–Czech Republic programme.

More information about the project is available at:

https://interreg.at-cz.eu/cz/katalog-projektu/2021-2027/vyzkum-a-inovace/atcz00060_ipmai

## Installation

Python 3.12 or newer is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the main launcher:

```bash
python run_0_main_launcher.py
```

## Licensing

### Source Code

All software source code contained in this repository is distributed under the **MIT License**.

The MIT License permits use, modification, redistribution, and commercial use of the software, provided that the original copyright notice and license text are retained.

See:

```text
LICENSE
```

for the full license text.

### Documentation

Documentation contained in this repository is distributed under the:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

https://creativecommons.org/licenses/by/4.0/

Users may copy, redistribute, adapt, and reuse the documentation for any purpose, including commercial use, provided that appropriate attribution is given.

### Third-Party Components

This repository may include or depend upon third-party libraries distributed under their respective licenses. Users are responsible for complying with the license terms of those components.

## Authors

CELS – Computational Engineering and Learning Systems

Faculty of Science

University of South Bohemia in České Budějovice

https://cels.prf.jcu.cz

## Citation

If you use this software in academic work, please cite the relevant publications associated with the methods implemented in this repository.
