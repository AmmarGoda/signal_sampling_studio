**Digital Signal Processing - Task 2**

---

## Overview

Sampling-Theory Studio is a Python application designed for **Digital Signal Processing (DSP) Task 2**. This tool enables users to compose, visualize, and interact with various sinusoidal signals. With an intuitive GUI, users can create complex signals by combining multiple sinusoids, save and load signal compositions, and visualize the resulting signals seamlessly.

---

## Key Features

- **Compose and Manage Sinusoidal Signals:**
  - Create complex signals by combining multiple sinusoids with adjustable frequency, amplitude, and phase.

- **Save and Load Signal Compositions:**
  - Save composed signals to CSV files and load them for further analysis.

- **Real-time Signal Visualization:**
  - Visualize the composed signal in real-time with an interactive plot.

- **Adjustable Noise Level:**
  - Add noise to the signal using a slider for more realistic signal analysis.

- **Sampling Frequency Control:**
  - Adjust the sampling frequency for signal processing.

---

## Application Interface

The graphical interface offers a clean and organized layout for streamlined user interaction.

#### Descriptions:
1. **Signal Composition Area:** Input fields for frequency, amplitude, and phase to compose sinusoids.
2. **Control Buttons:**
   - Add Sinusoid button to add the current sinusoid to the composition.
   - Save and Load button to save the composed signal and load it for analysis.
   - Load Only button to load the composed signal without saving.
3. **Interactive Controls:**
   - Noise slider to adjust the noise level in the signal.
   - Sampling frequency input to set the desired sampling frequency.
4. **Signal List:** Display the list of added sinusoids.
5. **Signal Plot:** Visualize the composed signal in real-time.

---

## How to Use

1. **Compose Signal:**
   - Input the desired frequency, amplitude, and phase for a sinusoid.
   - Click the 'Add Sinusoid' button to add it to the composition.
2. **Save and Load Signal:**
   - Click the 'Save and Load' button to save the composed signal to a CSV file and load it for analysis.
3. **Load Signal:**
   - Click the 'Load Only' button to load the composed signal without saving.
4. **Adjust Noise Level:**
   - Use the noise slider to add noise to the signal.
5. **Set Sampling Frequency:**
   - Input the desired sampling frequency in the sampling frequency input field.
6. **Visualize Signal:**
   - The composed signal will be visualized in the signal plot area in real-time.

---

## Installation

### Prerequisites

- Python 3.8+
- Required Libraries:
  - PyQt5 or PySide2 (for GUI)
  - NumPy
  - pandas
  - pyqtgraph

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/AmmarGoda/signal_sampling_studio.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python sampling_theory_studio.py
   ```

---

## Acknowledgments

This project is part of the **Digital Signal Processing** course. Special thanks to the course instructors and team members for their guidance and support.
