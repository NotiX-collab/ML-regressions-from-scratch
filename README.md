# ML Regressions from Scratch: The Power of Vectorization 🚀

This project is a deep dive into the mechanics of Machine Learning, documenting my transition from basic mathematical loops to high-performance vectorized implementations.

### 🧠 Core Concept
The goal was to implement **Linear** and **Logistic Regressions** using only `NumPy` and `Matplotlib`. No `scikit-learn`, no high-level black boxes, just pure math and Python.

### 📈 Performance Breakthrough
By moving from explicit Python loops to **Vectorized Operations** (Linear Algebra approach), I achieved massive performance gains:
* **Linear Regression:** ~200x faster implementation.
* **Logistic Regression:** ~100x faster implementation.

### 🛠 Key Features
* **Gradient Descent:** Implemented from scratch (both loop-based and vectorized versions).
* **Numerical Stability:** Custom Log-Loss function with epsilon-clipping to prevent logarithmic singularity (handling $log(0)$ cases).
* **Comparative Analysis:** Real-time plotting of cost functions and execution time differences.
* **Pure NumPy:** Leveraging matrix-vector multiplication to handle datasets efficiently.

### 🧪 Philosophy: Under the Hood
I am deeply convinced that to become a true master in any field, one must understand exactly how things work "under the hood."

Building these models from scratch isn't about reinventing the wheel—it's about gaining the fundamental insight that you simply cannot get by using high-level libraries. Mastery comes from the ability to deconstruct a complex system into its core components and rebuild it with your own hands.
