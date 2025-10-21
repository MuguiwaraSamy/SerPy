# 🌀 SerPy — Statistical single-exposure Retrieval of directional dark-field in PYthon

**SerPy** implements a **statistical single-exposure framework** for retrieving **transmission**, **isotropic dark-field**, and **directional dark-field (DF)** signals from a *single* pair of images (reference and sample).  
It is compatible with both **modulation-based** and **speckle-based** X-ray imaging techniques.

---

## 🚀 Overview

Conventional dark-field imaging requires multiple phase steps or modulator positions to separate phase, absorption, and scattering information.  
**SerPy** eliminates this need by relying on **local contrast statistics** computed within small sliding windows.  
From these statistics, the method reconstructs the **directional scattering ellipse** at each pixel, yielding quantitative maps of:

- **Transmission** \( T(\mathbf{r}) \)
- **Isotropic dark-field** \( D_f(\mathbf{r}) \)
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

```bash
git clone https://github.com/MuguiwaraSamy/SerPy.git
cd SerPy
pip install -U pip
pip install -e .
```
---

## ⚠️ Patent Notice

A patent application covering the underlying method and algorithm has been filed by the author(s):
**"Procédé d’imagerie et produit programme d’ordinateur correspondant"**, filed on September 26, 2025.

The publication of this code under the MIT license does **not** grant any license or rights
to practice the patented invention.  
Use of this software for **research and non-commercial purposes** is permitted under the terms of the MIT License.  
Any commercial use of the patented method requires a separate license from the patent holder.

