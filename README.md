# 🌀 SerPy — Statistical single-exposure Retrieval of directional dark-field in PYthon

[![PyPI version](https://img.shields.io/pypi/v/serpy_x.svg?color=3776AB&label=PyPI&logo=pypi)](https://pypi.org/project/serpy_x/)
[![Python versions](https://img.shields.io/pypi/pyversions/serpy_x.svg?logo=python&logoColor=white)](https://pypi.org/project/serpy_x/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/MuguiwaraSamy/SerPy)

---


**SerPy** implements a **statistical single-exposure framework** for retrieving **transmission**, **isotropic dark-field**, and **directional dark-field (DF)** signals from a *single* pair of images (reference and sample).  
It is compatible with both **modulation-based**, **speckle-based**, **single grid**, **edge illumination**, and **grating-based** X-ray imaging techniques.

---

## 🚀 Overview

Conventional dark-field imaging requires multiple phase steps or modulator positions to separate phase, absorption, and scattering information.  
**SerPy** eliminates this need by relying on **local contrast statistics** computed within small sliding windows.  
From these statistics, the method reconstructs the **directional scattering ellipse** at each pixel, yielding quantitative maps of:

- **Transmission** $ T(\mathbf{r})$
- **Isotropic dark-field** $ D_f(\mathbf{r}) $
- **Directional dark-field**: orientation, anisotropy, and magnitude

---

## ✳️ Key Features

- 🔹 **Single-exposure** retrieval — no phase stepping or mask motion required  
- 🔹 **Local statistical analysis** of contrast anisotropy  
- 🔹 **Directional dark-field extraction** via ellipse fitting in polar space  
- 🔹 **Orientation smoothing** using circular Gaussian filtering  
- 🔹 **Vectorized & Numba-accelerated** kernels for high performance  
- 🔹 **Compatible with** modulation-based, speckle-based, single grid, edge illumination, grating-based setups  

---



## ⚙️ Installation

⚠️ **Important — Create a dedicated Python environment**

`SerPy-X` relies on `numba`, which is **not yet compatible with Python 3.14**.  
Therefore, it is **strongly recommended** to install SerPy-X inside a dedicated environment  
using a compatible Python version (≥3.9 and <3.14), for example Python 3.12.

---

### ▶️ Option 1 — Install from PyPI (recommended)

#### 🔹 Using conda
```bash
conda create -n serpy-env python=3.12
conda activate serpy-env

pip install serpy_x
```

### 🔹 Using venv (standard virtual environment)
```bash
python3.12 -m venv serpy-env
source serpy-env/bin/activate

pip install --upgrade pip
pip install serpy_x
``` 
### ▶️ From source
```bash
git clone https://github.com/MuguiwaraSamy/SerPy.git
cd SerPy

# Create a compatible Python environment first
conda create -n serpy-dev python=3.12
conda activate serpy-dev

pip install --upgrade pip
pip install -e .    
```
---

## 🧠 Quickstart Example

```python
import numpy as np
from Serpy.statdf import retrieval_Algorithm

# img, ref : 2D numpy arrays (sample and reference)
out = retrieval_Algorithm(img, ref, n_angles=19, window_size=5)
```
---
| Key | Description |
|-----|--------------|
| `angles` | Array of sampled orientation angles (in radians) used for the directional analysis |
| `T` | Transmission map \( T(\mathbf{r}) \) |
| `Non-oriented Df` | Isotropic dark-field (scalar visibility loss) |
| `Oriented Df` | Directional dark-field profiles sampled along each orientation \((H, W, N_\theta)\) |
| `major axis` | Semi-major axis of the fitted ellipse (maximum scattering direction) |
| `minor axis` | Semi-minor axis of the fitted ellipse (minimum scattering direction) |
| `Orientation` | Raw ellipse orientation (principal scattering direction, before smoothing) |
| `Corrected Orientation` | Smoothed and π-periodic orientation map |
| `saturation` | Local angular coherence or stability of the orientation field |
| *(optionally)* `mean_s`, `std_s`, `mean_r`, `std_r` | Intermediate directional statistics (sample/reference means and standard deviations), included if `return_intermediates=True` |
---

## 📘 Full Example Notebook

A complete, end-to-end demonstration of the SerPy-X pipeline is provided in: 
`/examples/exemples.ipynb`

---

## ⚠️ Patent Notice

A patent application covering the underlying method and algorithm has been filed by the author(s):
**"Procédé d’imagerie et produit programme d’ordinateur correspondant"**, filed on September 26, 2025.

The publication of this code under the MIT license does **not** grant any license or rights
to practice the patented invention.  
Use of this software for **research and non-commercial purposes** is permitted under the terms of the MIT License.  
Any commercial use of the patented method requires a separate license from the patent holder.

⸻

📜 License

Distributed under the MIT License.
© 2025 Samy Kefs. All rights reserved.

⸻