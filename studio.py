import sys
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QMainWindow, QSlider, QDialog, QFormLayout, QSpinBox, QLabel, QFileDialog, QComboBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
import numpy as np
import pandas as pd

class ComposeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compose Signal")
        self.resize(600, 600)  # Increased height to accommodate the plot

        # Layout setup
        self.layout = QVBoxLayout(self)
        self.signals_layout = QFormLayout()

        # Store components of the signal (frequency, amplitude, phase)
        self.sinusoids = []
        self.max_frequency = 0

        # Frequency input with QSpinBox
        self.frequency_input = QSpinBox()
        self.frequency_input.setRange(1, 100)  # Frequency range 1 to 100 Hz
        self.frequency_input.setSuffix(" Hz")

        # Amplitude input with QSpinBox
        self.amplitude_input = QSpinBox()
        self.amplitude_input.setRange(1, 100)  # Amplitude range 1 to 100

        # Phase input with QDoubleSpinBox
        self.phase_input = QDoubleSpinBox()
        self.phase_input.setRange(-360, 360)  # Phase range in degrees
        self.phase_input.setSuffix("°")
        self.phase_input.setSingleStep(1.0)

        # Add Sinusoid button
        self.add_signal_button = QPushButton("Add Sinusoid")
        self.add_signal_button.clicked.connect(self.add_sinusoid)

        # Save and Load button
        self.save_button = QPushButton("Save and Load")
        self.save_button.clicked.connect(self.save_and_load)

        # Load Only button
        self.load_button = QPushButton("Load Only")
        self.load_button.clicked.connect(self.load_only)

        # Adding widgets to layout
        self.signals_layout.addRow("Frequency:", self.frequency_input)
        self.signals_layout.addRow("Amplitude:", self.amplitude_input)
        self.signals_layout.addRow("Phase:", self.phase_input)
        self.layout.addLayout(self.signals_layout)
        self.layout.addWidget(self.add_signal_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.load_button)

        # Plot widget for visualizing the composed signal
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getAxis('left').setTicks([])
        self.plot_widget.getAxis('bottom').setTicks([])
        self.plot_widget.getAxis('left').setStyle(showValues=False)
        self.plot_widget.getAxis('bottom').setStyle(showValues=False)
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.XYAxes)
        self.layout.addWidget(self.plot_widget)

        # List widget for displaying added sinusoids
        self.sinusoids_list = QListWidget()
        self.layout.addWidget(self.sinusoids_list)

    def add_sinusoid(self):
        # Capture values from input widgets
        freq = self.frequency_input.value()
        amp = self.amplitude_input.value()
        phase = np.radians(self.phase_input.value())  # Convert phase from degrees to radians
        self.sinusoids.append((freq, amp, phase))

        # Update max frequency
        self.max_frequency = max(self.max_frequency, freq)

        # Add to list widget
        item = QListWidgetItem(f"Frequency={freq}Hz, Amplitude={amp}, Phase={self.phase_input.value()}°")
        self.sinusoids_list.addItem(item)

        # Update the plot with the new composed signal
        self.update_plot()

    def save_and_load(self):
        # Generate signal
        signal = self.generate_composed_signal()

        # Prompt user to choose file location and name
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Composed Signal", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            # Save to CSV with max frequency as metadata
            with open(file_path, 'w') as file:
                # Write max frequency as a comment line at the top
                file.write(f"# MaxFrequency: {self.max_frequency}\n")
                df = pd.DataFrame(signal, columns=['Time', 'Amplitude'])
                df.to_csv(file, index=False)
            
            self.accept()
            return signal, self.max_frequency
        else:
            return None, None

    def load_only(self):
        # Generate signal without saving, then return signal for plotting
        signal = self.generate_composed_signal()
        self.accept()
        return signal

    def generate_composed_signal(self):
        # Generate high-resolution signal (1000 points)
        t = np.linspace(0, 1, 1000)
        signal = np.zeros_like(t)
        for freq, amp, phase in self.sinusoids:
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        return np.column_stack((t, signal))

    def update_plot(self):
        # Generate the composed signal
        composed_signal = self.generate_composed_signal()
        t, signal = composed_signal[:, 0], composed_signal[:, 1]

        # Clear the plot and plot the new signal
        self.plot_widget.clear()
        self.plot_widget.plot(t, signal, pen='b', name="Original Signal")

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Sampling-Theory Studio")
        self.setGeometry(100, 100, 1200, 800)

        self.init_ui()

        self.original_signal = None
        self.fmax = 1 # Default Value
    
    def init_ui(self):
        self.setWindowIcon(QIcon())
        # Main widget setup
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        main_layout = QVBoxLayout(self.main_widget)

        # Control panel in a group box
        control_group_box = QWidget(self)
        control_layout = QVBoxLayout(control_group_box)
        # control_group_box.setStyleSheet("background-color: #999999; border-radius: 10px; padding: 10px;")
        main_layout.addWidget(control_group_box)

        # Top control layout
        top_control_layout = QHBoxLayout()
        main_layout.addLayout(top_control_layout)

        # Load file button
        self.load_file_button = QPushButton("Load", self)
        self.load_file_button.setToolTip("Load Signal Using CSV File")
        self.load_file_button.clicked.connect(self.load_csv_file)
        top_control_layout.addWidget(self.load_file_button)

        # Compose button
        self.compose_button = QPushButton("Compose", self)
        self.compose_button.setToolTip("Compose Signal")
        self.compose_button.clicked.connect(self.open_compose_dialog)
        top_control_layout.addWidget(self.compose_button)

        # Noise Slider
        noise_label = QLabel("Noise:", self)
        top_control_layout.addWidget(noise_label)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 50)  # Adjusted range for more noticeable noise
        self.noise_slider.setValue(0)
        self.noise_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_slider.setTickInterval(5)
        self.noise_slider.setSingleStep(1)
        self.noise_slider.valueChanged.connect(self.update_signal_with_noise)
        top_control_layout.addWidget(self.noise_slider)


        # Sampling Frequency Input
        sampling_freq_label = QLabel("Sampling Frequency:", self)
        top_control_layout.addWidget(sampling_freq_label)
        self.sampling_freq_input = QSpinBox()
        self.sampling_freq_input.setRange(1, 1000)  # Frequency range 1 to 1000 Hz
        self.sampling_freq_input.setSuffix(" Hz")
        self.sampling_freq_input.setValue(10)
        self.sampling_freq_input.setSingleStep(1)  # Set step size to 10
        self.sampling_freq_input.valueChanged.connect(self.update_sampled_signal)
        top_control_layout.addWidget(self.sampling_freq_input)

        # Display Mode ComboBox
        display_mode_label = QLabel("Display Mode:", self)
        top_control_layout.addWidget(display_mode_label)
        self.display_mode_combo = QComboBox(self)
        self.display_mode_combo.addItems(["Absolute", "Normalized"])
        self.display_mode_combo.currentIndexChanged.connect(self.update_display_mode)
        top_control_layout.addWidget(self.display_mode_combo)

        # Reconstruction Method ComboBox
        reconstruction_label = QLabel("Reconstruction Method:", self)
        top_control_layout.addWidget(reconstruction_label)

        self.reconstruction_combo = QComboBox(self)
        self.reconstruction_combo.addItems(["Whittaker-Shannon", "Nearest Neighbor", "Linear Interpolation"])
        self.reconstruction_combo.currentIndexChanged.connect(self.update_reconstruction_method)
        top_control_layout.addWidget(self.reconstruction_combo)

        # Graph widgets with updated layout
        self.graph1 = pg.PlotWidget()
        self.graph1.setLabel('bottom', 'Time', 's')
        self.graph1.setLabel('left', 'Amplitude', 'V')
        self.graph1.showGrid(x=True, y=True, alpha=0.5)
        self.graph1.setTitle("Original Signal")
        self.style_plot_widget(self.graph1)
        main_layout.addWidget(self.graph1)

        self.graph2 = pg.PlotWidget()
        self.graph2.setLabel('bottom', 'Time', 's')
        self.graph2.setLabel('left', 'Amplitude', 'V')
        self.graph2.showGrid(x=True, y=True, alpha=0.5)
        self.graph2.setTitle("Reconstructed Signal")
        self.style_plot_widget(self.graph2)
        main_layout.addWidget(self.graph2)

        self.graph3 = pg.PlotWidget()
        self.graph3.setLabel('bottom', 'Time', 's')
        self.graph3.setLabel('left', 'Amplitude', 'V')
        self.graph3.showGrid(x=True, y=True, alpha=0.5)
        self.graph3.setTitle("Difference in Time Domain")
        self.style_plot_widget(self.graph3)
        main_layout.addWidget(self.graph3)

        self.graph4 = pg.PlotWidget()
        self.graph4.setLabel('bottom', 'Frequency', 'Hz')
        self.graph4.setLabel('left', 'Magnitude')
        self.graph4.showGrid(x=True, y=True, alpha=0.5)
        self.graph4.setTitle("Difference in Frequency Domain")
        self.style_plot_widget(self.graph4)
        main_layout.addWidget(self.graph4)

    def open_compose_dialog(self):
        dialog = ComposeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.original_signal = dialog.save_and_load() if dialog.save_button.isChecked() else dialog.load_only()
            t, signal = self.original_signal[:, 0], self.original_signal[:, 1]
                    
            # Update fmax in MainWindow to reflect the max_frequency of the composed signal
            max_frequency = dialog.max_frequency
            self.fmax = max_frequency
            self.graph1.clear()
            self.graph1.setTitle(f"Original Signal (fmax = {self.fmax} Hz)")
            self.graph1.plot(t, signal, pen=pg.mkPen(color='b', width=1), name="Original Signal")
            self.update_sampled_signal()

    def load_csv_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV Signal File", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            # Read max frequency from the first line (metadata comment)
            with open(file_path, 'r') as file:
                first_line = file.readline().strip()
                if first_line.startswith("# MaxFrequency:"):
                    self.fmax = float(first_line.split(":")[1].strip())
                else:
                    self.fmax = None  # Handle missing metadata if necessary

            # Read the actual signal data
            df = pd.read_csv(file_path, comment='#')  # Skip the metadata comment line when reading


            if 'Time' in df.columns and 'Amplitude' in df.columns:
                t = df['Time'].values
                signal = df['Amplitude'].values
                self.original_signal = np.column_stack((t, signal))
                self.graph1.clear()
                self.graph1.setTitle(f"Original Signal (fmax = {self.fmax} Hz)")
                self.graph1.plot(t, signal, pen=pg.mkPen(color='b', width=1), name="Original Signal")
                self.update_sampled_signal()
    
    def update_display_mode(self):
        mode = self.display_mode_combo.currentText()
        if mode == "Absolute":
            self.sampling_freq_input.setRange(1, 1000)
            self.sampling_freq_input.setValue(10)
            self.sampling_freq_input.setSuffix(" Hz")
        elif mode == "Normalized":
            self.sampling_freq_input.setRange(1, 4)
            self.sampling_freq_input.setValue(1)
            self.sampling_freq_input.setSingleStep(1)
            self.sampling_freq_input.setSuffix(" x fmax")

    def add_noise_to_signal(self, signal, snr_db):
        # Calculate signal power and noise power
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
        
        noise *= 2

        # Add noise to the signal
        noisy_signal = signal + noise
        return noisy_signal

    def update_signal_with_noise(self):
        """Applies noise to the original signal and then updates the sampled view."""
        if self.original_signal is not None:
            t, signal = self.original_signal[:, 0], self.original_signal[:, 1]
            
            # Calculate SNR and add noise
            max_snr_db = self.noise_slider.maximum()
            snr_db = max_snr_db - self.noise_slider.value()  # Invert the slider value
            self.noisy_signal = self.add_noise_to_signal(signal, snr_db)
            
            # Plot noisy signal for reference
            self.graph1.clear()
            self.graph1.plot(t, self.noisy_signal, pen=pg.mkPen(color='r', width=1), name="Noisy Signal")

            # Update sampled view
            self.update_sampled_signal()  # Reapply sampling on the noisy signal

    def sample_signal(self, signal, sampling_freq):
        t, amplitude = signal[:, 0], signal[:, 1]
        # Calculate sampling interval and add handling for zero division
        sampling_interval = max(1, len(t) // int(sampling_freq)) if sampling_freq > 0 else 1
        sampled_indices = np.arange(0, len(t), sampling_interval)
        sampled_t = t[sampled_indices]
        sampled_amplitude = amplitude[sampled_indices]
        return sampled_t, sampled_amplitude

    def update_sampled_signal(self):
        """Applies sampling to the signal (either original or noisy) and displays the result."""
        if self.original_signal is not None:
            # Retrieve current mode and determine sampling frequency

            mode = self.display_mode_combo.currentText()
            if mode == "Absolute":
                self.sampling_freq = self.sampling_freq_input.value()  # Use input directly in Hz
            elif mode == "Normalized":
                # Calculate sampling frequency as a multiple of fmax
                self.sampling_freq = (self.sampling_freq_input.value() * self.fmax + 1)

            # Check for valid sampling frequency to prevent issues
            if self.sampling_freq <= 0 or self.sampling_freq > len(self.original_signal) / (self.original_signal[-1, 0] - self.original_signal[0, 0]):
                print("Invalid sampling frequency.")
                return

            # Choose whether to sample the original or noisy signal
            t = self.original_signal[:, 0]
            signal = self.noisy_signal if hasattr(self, 'noisy_signal') else self.original_signal[:, 1]
            
            # Apply sampling based on determined sampling frequency
            self.sampled_t, self.sampled_amplitude = self.sample_signal(np.column_stack((t, signal)), self.sampling_freq)

            # Plot the original or noisy signal with sampling markers
            self.graph1.clear()
            self.graph1.plot(t, signal, pen=pg.mkPen(color='b', width=1), name="Signal with Sampling")
            
            # Plot the sampled points with an optional offset
            # y_offset = -5.5 * np.max(np.abs(signal))  # Adjust the offset as needed
            # self.graph1.plot(self.sampled_t, self.sampled_amplitude + y_offset, pen=None, symbol='s', symbolBrush='b', symbolSize=10, name="Sampled Points")
            self.graph1.plot(self.sampled_t, self.sampled_amplitude, pen=None, symbol='o', symbolBrush='b', symbolSize=6, name="Sampled Points")

            # Reconstruct the signal
            self.reconstruct_signal()

            # Calculate and plot the difference
            self.plot_difference()

            self.plot_frequency_domain()
            

    def update_reconstruction_method(self):
        self.update_sampled_signal()
        
    def reconstruct_signal(self):
        if self.original_signal is None:
            return

        t = self.original_signal[:, 0]

        method = self.reconstruction_combo.currentText()
        if method == "Whittaker-Shannon":
            self.reconstructed_signal = self.reconstruct_whittaker_shannon()
        elif method == "Nearest Neighbor":
            self.reconstructed_signal = self.reconstruct_nearest_neighbor()
        elif method == "Linear Interpolation":
            self.reconstructed_signal = self.reconstruct_linear_interpolation()

        # Clear the plot and display the reconstructed signal
        self.graph2.clear()
        self.graph2.plot(t, self.reconstructed_signal, pen=pg.mkPen(color='r', width=1), name="Reconstructed Signal")
        # self.graph2.plot(self.sampled_t, self.sampled_amplitude, pen=None, symbol='o', symbolBrush='b', symbolSize=6, name="Sampled Points")

    def reconstruct_whittaker_shannon(self):
        t = self.original_signal[:, 0]
        
        def sinc_interp(x, s, u, fs):
            """
            Sinc interpolation to reconstruct the continuous signal.
            x - the time points to evaluate the interpolated signal
            s - the sampled signal values
            u - the sample points
            fs - the sampling frequency
            """
            return np.array([np.sum(s * np.sinc(fs * (xi - u))) for xi in x])
        
        reconstructed_signal = sinc_interp(t, self.sampled_amplitude, self.sampled_t, self.sampling_freq)
        return reconstructed_signal
    
    def reconstruct_nearest_neighbor(self):
        t = self.original_signal[:, 0]
        reconstructed_signal = np.zeros_like(t)
        # Iterate over each time point in t
        for i, ti in enumerate(t):
            # If ti is before the first sample time, use the first sample value
            if ti <= self.sampled_t[0]:
                reconstructed_signal[i] = self.sampled_amplitude[0]
            # Otherwise, find the last sampled point before ti
            else:
                # Find the index of the largest sampled time less than or equal to ti
                idx = np.searchsorted(self.sampled_t, ti) - 1
                reconstructed_signal[i] = self.sampled_amplitude[idx]
        return reconstructed_signal

    def reconstruct_linear_interpolation(self):
        t = self.original_signal[:, 0]
        reconstructed_signal = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti <= self.sampled_t[0]:
                reconstructed_signal[i] = self.sampled_amplitude[0]
            elif ti >= self.sampled_t[-1]:
                reconstructed_signal[i] = self.sampled_amplitude[-1]
            else:
                idx = np.searchsorted(self.sampled_t, ti) - 1
                t1, t2 = self.sampled_t[idx], self.sampled_t[idx + 1]
                a1, a2 = self.sampled_amplitude[idx], self.sampled_amplitude[idx + 1]
                reconstructed_signal[i] = a1 + (a2 - a1) * (ti - t1) / (t2 - t1)
        return reconstructed_signal
    
    # def plot_difference(self):
    #     if self.original_signal is None or self.reconstructed_signal is None:
    #         return

    #     t = self.original_signal[:, 0]
    #     signal = self.noisy_signal if hasattr(self, 'noisy_signal') else self.original_signal[:, 1]
    #     self.graph3.clear()
    #     self.graph3.plot(t, signal, pen=pg.mkPen(color='b', width=1), name="Original Signal")
    #     self.graph3.plot(t, self.reconstructed_signal, pen=pg.mkPen(color='r', width=1), name="Reconstructed Signal")
    
    def plot_difference(self):    
        if self.original_signal is None or self.reconstructed_signal is None:
            return

        # Extract time and signal values
        t = self.original_signal[:, 0]
        original_signal_values = self.original_signal[:, 1]
        reconstructed_signal_values = self.reconstructed_signal

        # Calculate the difference between original and reconstructed signals
        difference_signal = original_signal_values - reconstructed_signal_values

        # Clear previous plots on graph3 and plot the difference signal
        self.graph3.clear()
        self.graph3.plot(t, difference_signal, pen=pg.mkPen(color='g', width=1), name="Difference Signal")
        
    def plot_frequency_domain(self):
        if self.original_signal is None:
            return

        # Extract time and signal values
        t = self.original_signal[:, 0]
        signal = self.noisy_signal if hasattr(self, 'noisy_signal') else self.original_signal[:, 1]

        # Sampling frequency (f_s)
        if hasattr(self, 'sampling_freq') and self.sampling_freq > 0:
            f_s = self.sampling_freq
        else:
            print("Sampling frequency not defined. Defaulting to signal's time step.")
            f_s = 1 / (t[1] - t[0])  # Approximate based on time step

        # Compute FFT and frequencies
        N = len(signal)
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, d=(t[1] - t[0]))
        fft_magnitude = np.abs(fft_signal)  # Take magnitude of FFT

        # Create extended periodic copies of the frequency domain
        num_periods = 3  # Number of total periods to display (adjust as needed)
        extended_freqs = []
        extended_fft = []

        # Repeat symmetrically to the left and right
        for k in range(-num_periods // 2, num_periods // 2 + 1):
            shifted_freqs = freqs + k * f_s  # Shift frequencies by multiples of f_s
            extended_freqs.append(shifted_freqs)
            extended_fft.append(fft_magnitude)  # Reuse the same FFT magnitude

        # Combine extended frequencies and FFT magnitudes
        extended_freqs = np.concatenate(extended_freqs)
        extended_fft = np.concatenate(extended_fft)

        # Sort frequencies and corresponding FFT values
        sorted_indices = np.argsort(extended_freqs)
        extended_freqs = extended_freqs[sorted_indices]
        extended_fft = extended_fft[sorted_indices]

        # Clear the graph and plot the periodic frequency spectrum
        self.graph4.clear()
        self.graph4.setTitle("Frequency Domain with Periodicity")
        self.graph4.plot(extended_freqs, extended_fft, pen=pg.mkPen(color='b', width=1), name="Periodic FFT")

        # Mark the Nyquist frequency
        nyquist_freq = f_s / 2
        self.graph4.addLine(x=nyquist_freq, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine), name="Nyquist Limit")
        self.graph4.addLine(x=-nyquist_freq, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))

    def style_plot_widget(self, plot_widget):
        plot_widget.getAxis('left').setPen('#2c3e50')  # Set axis color
        plot_widget.getAxis('bottom').setPen('#2c3e50')
        plot_widget.getAxis('left').setStyle(tickTextOffset=10, showValues=True)
        plot_widget.getAxis('bottom').setStyle(tickTextOffset=10, showValues=True)
        plot_widget.showGrid(x=True, y=True, alpha=0.3)  # Add grid lines with transparency

def apply_style(app):
    style = style = """
            /* General App Styling */
            QWidget {
                background-color: #e8ecf1;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #2c3e50;
            }
            QMainWindow {
                background-color: #ffffff;
                border: 1px solid #dcdde1;
                border-radius: 8px;
                padding: 10px;
            }

            /* Buttons */
            QPushButton {
                background-color: #1abc9c;
                color: white;
                border: 2px solid #16a085;
                border-radius: 12px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #16a085;
                border-color: #149174;
            }
            QPushButton:pressed {
                background-color: #149174;
                border-color: #12876b;
            }
            
            /* ComboBox */
            QComboBox {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 5px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }

            /* SpinBox */
            QSpinBox {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 5px;
            }      
            /* Sliders */
            QSlider::groove:horizontal {
                background: #bdc3c7;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #1abc9c;
                width: 18px;
                height: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
            QSlider::groove:vertical {
                background: #bdc3c7;
                width: 10px;
                border-radius: 5px;
            }
            QSlider::handle:vertical {
                background: #1abc9c;
                height: 18px;
                width: 18px;
                margin: 0 -4px;
                border-radius: 9px;
            }
            
            /* List Widget */
            QListWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                padding: 5px;
            }

            /* Labels */
            QLabel {
                color: #34495e;
                font-weight: 500;
                padding-bottom: 5px;
            }

            /* Plot Widgets */
            QPlotWidget {
                border: 2px solid #1abc9c;
                border-radius: 8px;
            }

            /* Custom Layout and MainWindow Styling */
            #MainWindow {
                background-color: #f7f9fa;
                border-radius: 10px;
                padding: 15px;
            }
            #TopControlLayout {
                background-color: #ffffff;
                border-bottom: 1px solid #ecf0f1;
                padding: 10px;
            }
            #MainLayout {
                padding: 15px;
            }
            #GraphLayout {
                background-color: #f7f9fa;
                border: 1px solid #dcdde1;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
            }

            /* Graph Styling */
            #Graph1, #Graph2, #Graph3, #Graph4 {
                border: 2px solid #1abc9c;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
            }
        """

    app.setStyleSheet(style)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_style(app)
    app.setWindowIcon(QIcon())
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
