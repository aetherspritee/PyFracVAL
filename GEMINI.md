# Gemini Workspace

This file provides context for the Gemini AI model to understand the project and assist in its development.

## Project: PyFracVAL

This project is a Python reimplementation of the Fortran-based FracVAL algorithm, which was developed by Moran et al. in their 2019 paper, "FracVAL: An improved tunable algorithm of cluster–cluster aggregation for generation of fractal structures formed by polydisperse primary particles." The original Fortran code is located in the `docs/FracVAL` directory, and the paper can be found in `docs/moran2019.pdf`.

The primary goal of this project is to create a stable and complete Python version of the FracVAL algorithm, while also extending its functionality and addressing any existing bugs.

### Core Objectives:

*   **Replicate FracVAL in Python:** The main objective is to faithfully reproduce the functionality of the original Fortran code in Python.
*   **Enhance Functionality:** The project also aims to extend the capabilities of the original algorithm, adding new features and improvements.
*   **Bug Fixes:** Any bugs or issues present in the original FracVAL implementation will be addressed and resolved.
*   **Stability and Completeness:** The final Python implementation should be stable, robust, and feature-complete.

### Key Technologies:

*   **Programming Language:** Python
*   **Libraries:** The specific libraries to be used will be determined during the development process, but they will likely include numerical and scientific computing libraries such as NumPy and SciPy, as well as data visualization libraries like Matplotlib or Plotly.

### Development Plan:

1.  **Analyze the Original Code:** The first step is to thoroughly analyze the Fortran code in `docs/FracVAL` to understand its structure, algorithms, and dependencies.
2.  **Translate to Python:** The Fortran code will then be translated into Python, with a focus on creating a clear, readable, and maintainable codebase.
3.  **Implement Core Functionality:** The core functionality of the FracVAL algorithm will be implemented in Python, including the generation of primary particles, cluster-cluster aggregation, and the calculation of fractal dimensions.
4.  **Develop a Testing Framework:** A comprehensive testing framework will be developed to ensure the correctness and stability of the Python implementation.
5.  **Extend Functionality:** Once the core functionality is in place, new features and improvements will be added, such as support for different particle size distributions, improved visualization options, and enhanced performance.
6.  **Documentation:** The final Python implementation will be thoroughly documented, with clear explanations of the code, algorithms, and usage.

This project will involve a thorough analysis of the original paper and Fortran code, followed by a careful and well-documented Python implementation. The end result will be a powerful and user-friendly tool for generating fractal structures, building upon the solid foundation of the original FracVAL algorithm.