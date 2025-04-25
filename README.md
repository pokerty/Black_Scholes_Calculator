# Black-Scholes Option Pricing Calculator

This app calculates the price of European call and put options using the Black-Scholes model. It also calculates and visualizes the option Greeks and provides heatmaps showing the option price sensitivity to changes in the underlying asset price and volatility.

## Features

- Calculate Call and Put option prices based on user inputs.
- Calculate and display the option Greeks: Delta, Gamma, Theta, Vega, Rho.
- Visualize how each Greek changes with the underlying asset price.
- Display heatmaps for Call and Put option prices based on varying underlying prices and volatilities.
- Interactive sidebar for adjusting input parameters (Risk-Free Rate, Underlying Price, Strike Price, Time to Expiry, Volatility).

## Setup

1.  **Clone the repository or download the script.**
2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Navigate to the project directory in your terminal.**
2.  **Run the Streamlit application:**
    ```bash
    streamlit run black-scholes.py
    ```
3.  **Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).**
4.  **Use the sidebar to adjust the input parameters and see the results update in real-time.**

## Disclaimer

This calculator is for educational purposes only and should not be used for making actual investment decisions. Financial markets involve risk, and the Black-Scholes model has its own limitations and assumptions.
