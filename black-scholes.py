import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd

# Set seaborn style globally
sns.set_theme(style="whitegrid")

def blackScholes(S, K, r, T, sigma, type="c"):
    "Calculate Black Scholes option price for a call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)

        return price
    except:  
        st.sidebar.error("Please confirm all option parameters!")


def optionDelta (S, K, r, T, sigma, type="c"):
    "Calculates option delta"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            delta = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta = -norm.cdf(-d1, 0, 1)

        return delta
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionGamma (S, K, r, T, sigma):
    "Calculates option gamma"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        gamma = norm.pdf(d1, 0, 1)/ (S * sigma * np.sqrt(T))
        return gamma
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionTheta(S, K, r, T, sigma, type="c"):
    "Calculates option theta"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2, 0, 1)

        elif type == "p":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
        return theta/365
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionVega (S, K, r, T, sigma):
    "Calculates option vega"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        vega = S * np.sqrt(T) * norm.pdf(d1, 0, 1) * 0.01
        return vega
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionRho(S, K, r, T, sigma, type="c"):
    "Calculates option rho"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            rho = 0.01 * -K * T * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
        return rho
    except:
        st.sidebar.error("Please confirm all option parameters!")


# --- Streamlit App Configuration ---
st.set_page_config(page_title="Black-Scholes Calculator", layout="wide")

st.markdown("<h1 style='text-align: center;'>Black-Scholes Option Pricing Model</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar for Inputs ---
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Adjust the parameters below to see how they affect the option price and Greeks.")

r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.000, max_value=1.000, step=0.001, value=0.030, format="%.3f")
S = st.sidebar.number_input("Underlying Asset Price (S)", min_value=1.00, step=0.10, value=30.00, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", min_value=1.00, step=0.10, value=50.00, format="%.2f")
days_to_expiry = st.sidebar.number_input("Time to Expiry (Days)", min_value=1, step=1, value=250)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.001, max_value=1.000, step=0.01, value=0.30, format="%.3f")
type_input = st.sidebar.selectbox("Option Type", ["Call", "Put"])

# Convert days to years and set option type character
T = days_to_expiry / 365.0
option_type = "c" if type_input == "Call" else "p"

# Separator for Heatmap Inputs
st.sidebar.markdown("---")
st.sidebar.subheader("Heatmap Inputs")

# Volatility Range Slider
vol_min_default, vol_max_default = max(0.01, sigma - 0.1), sigma + 0.1 # Default range around current sigma
volatility_range = st.sidebar.slider(
    'Volatility Range (σ)',
    min_value=0.01,
    max_value=1.5, # Allow wider range exploration
    value=(vol_min_default, vol_max_default),
    step=0.01,
    format="%.2f"
)
vol_min_selected, vol_max_selected = volatility_range

# Spot Price Range Inputs
spot_min_default = S * 0.8
spot_max_default = S * 1.2
spot_min = st.sidebar.number_input('Min Spot Price ($)', value=spot_min_default, format="%.2f")
spot_max = st.sidebar.number_input('Max Spot Price ($)', value=spot_max_default, format="%.2f")


# --- Calculations ---
# Calculate current prices and Greeks
current_call_price = blackScholes(S, K, r, T, sigma, type="c")
current_put_price = blackScholes(S, K, r, T, sigma, type="p")
current_delta = optionDelta(S, K, r, T, sigma, type=option_type)
current_gamma = optionGamma(S, K, r, T, sigma)
current_theta = optionTheta(S, K, r, T, sigma, type=option_type)
current_vega = optionVega(S, K, r, T, sigma)
current_rho = optionRho(S, K, r, T, sigma, type=option_type)

# Calculate Greeks over a range of spot prices for plotting
spot_price_range = np.linspace(max(0.1, S - S*0.5), S + S*0.5, 100) # Generate a range around the current price

prices_range = [blackScholes(i, K, r, T, sigma, option_type) for i in spot_price_range]
deltas_range = [optionDelta(i, K, r, T, sigma, option_type) for i in spot_price_range]
gammas_range = [optionGamma(i, K, r, T, sigma) for i in spot_price_range]
thetas_range = [optionTheta(i, K, r, T, sigma, option_type) for i in spot_price_range]
vegas_range = [optionVega(i, K, r, T, sigma) for i in spot_price_range]
rhos_range = [optionRho(i, K, r, T, sigma, option_type) for i in spot_price_range]

# --- Display Results ---
st.subheader("Calculated Option Prices")
col1, col2 = st.columns(2)
col1.metric("Call Option Price", f"{current_call_price:.3f}" if current_call_price is not None else "N/A")
col2.metric("Put Option Price", f"{current_put_price:.3f}" if current_put_price is not None else "N/A")

st.subheader(f"Calculated Greeks for {type_input} Option")
gcol1, gcol2, gcol3 = st.columns(3)
gcol1.metric("Delta (Δ)", f"{current_delta:.3f}" if current_delta is not None else "N/A")
gcol2.metric("Gamma (Γ)", f"{current_gamma:.3f}" if current_gamma is not None else "N/A")
gcol3.metric("Theta (Θ)", f"{current_theta:.3f}" if current_theta is not None else "N/A", help="Per day")

gcol4, gcol5, gcol6 = st.columns(3)
gcol4.metric("Vega", f"{current_vega:.3f}" if current_vega is not None else "N/A", help="Per 1% change in Vol")
gcol5.metric("Rho (ρ)", f"{current_rho:.3f}" if current_rho is not None else "N/A", help="Per 1% change in r")
gcol6.metric("-", "") # Placeholder for alignment if needed

st.markdown("---")
st.subheader("Greeks Visualization")
st.markdown(f"Visualizing the Greeks for the selected **{type_input}** option across different underlying asset prices.")

# --- Plotting ---
# Create figures
fig_price, ax_price = plt.subplots(figsize=(8, 5))
sns.lineplot(x=spot_price_range, y=prices_range, ax=ax_price)
ax_price.set_title(f'{type_input} Option Price vs. Underlying Price')
ax_price.set_xlabel("Underlying Asset Price (S)")
ax_price.set_ylabel("Option Price")
ax_price.axvline(S, color='r', linestyle='--', label=f'Current S = {S:.2f}')
ax_price.legend()
fig_price.tight_layout()

fig_delta, ax_delta = plt.subplots(figsize=(8, 5))
sns.lineplot(x=spot_price_range, y=deltas_range, ax=ax_delta)
ax_delta.set_title('Delta (Δ) vs. Underlying Price')
ax_delta.set_xlabel("Underlying Asset Price (S)")
ax_delta.set_ylabel("Delta")
ax_delta.axvline(S, color='r', linestyle='--', label=f'Current S = {S:.2f}')
ax_delta.legend()
fig_delta.tight_layout()

fig_gamma, ax_gamma = plt.subplots(figsize=(8, 5))
sns.lineplot(x=spot_price_range, y=gammas_range, ax=ax_gamma)
ax_gamma.set_title('Gamma (Γ) vs. Underlying Price')
ax_gamma.set_xlabel("Underlying Asset Price (S)")
ax_gamma.set_ylabel("Gamma")
ax_gamma.axvline(S, color='r', linestyle='--', label=f'Current S = {S:.2f}')
ax_gamma.legend()
fig_gamma.tight_layout()

fig_theta, ax_theta = plt.subplots(figsize=(8, 5))
sns.lineplot(x=spot_price_range, y=thetas_range, ax=ax_theta)
ax_theta.set_title('Theta (Θ) vs. Underlying Price')
ax_theta.set_xlabel("Underlying Asset Price (S)")
ax_theta.set_ylabel("Theta (per day)")
ax_theta.axvline(S, color='r', linestyle='--', label=f'Current S = {S:.2f}')
ax_theta.legend()
fig_theta.tight_layout()

fig_vega, ax_vega = plt.subplots(figsize=(8, 5))
sns.lineplot(x=spot_price_range, y=vegas_range, ax=ax_vega)
ax_vega.set_title('Vega vs. Underlying Price')
ax_vega.set_xlabel("Underlying Asset Price (S)")
ax_vega.set_ylabel("Vega (per 1% vol change)")
ax_vega.axvline(S, color='r', linestyle='--', label=f'Current S = {S:.2f}')
ax_vega.legend()
fig_vega.tight_layout()

fig_rho, ax_rho = plt.subplots(figsize=(8, 5))
sns.lineplot(x=spot_price_range, y=rhos_range, ax=ax_rho)
ax_rho.set_title('Rho (ρ) vs. Underlying Price')
ax_rho.set_xlabel("Underlying Asset Price (S)")
ax_rho.set_ylabel("Rho (per 1% rate change)")
ax_rho.axvline(S, color='r', linestyle='--', label=f'Current S = {S:.2f}')
ax_rho.legend()
fig_rho.tight_layout()

# Display plots directly on the page
st.pyplot(fig_price)
st.pyplot(fig_delta)
st.pyplot(fig_gamma)
st.pyplot(fig_theta)
st.pyplot(fig_vega)
st.pyplot(fig_rho)

# --- Heatmap Calculation and Plotting ---
st.markdown("---")
st.subheader("Option Price Heatmaps")
st.markdown("Visualizing option price sensitivity to changes in Underlying Price and Volatility.")

# Define ranges for heatmap axes
heatmap_spot_range = np.linspace(spot_min, spot_max, 10) # Fewer points for performance
heatmap_vol_range = np.linspace(vol_min_selected, vol_max_selected, 10)

# Create grid for calculations
spot_grid, vol_grid = np.meshgrid(heatmap_spot_range, heatmap_vol_range)

# Calculate option prices for each point in the grid for BOTH Calls and Puts
call_heatmap_prices = np.zeros_like(spot_grid)
put_heatmap_prices = np.zeros_like(spot_grid)

for i in range(spot_grid.shape[0]):
    for j in range(spot_grid.shape[1]):
        # Calculate Call price
        call_price = blackScholes(spot_grid[i, j], K, r, T, vol_grid[i, j], type="c")
        call_heatmap_prices[i, j] = call_price if call_price is not None else np.nan

        # Calculate Put price
        put_price = blackScholes(spot_grid[i, j], K, r, T, vol_grid[i, j], type="p")
        put_heatmap_prices[i, j] = put_price if put_price is not None else np.nan

# Create DataFrames for easier plotting with labels
call_heatmap_df = pd.DataFrame(call_heatmap_prices,
                               index=[f"{vol:.2f}" for vol in heatmap_vol_range],
                               columns=[f"{spot:.2f}" for spot in heatmap_spot_range])
call_heatmap_df.index.name = 'Volatility (σ)'
call_heatmap_df.columns.name = 'Underlying Price (S)'

put_heatmap_df = pd.DataFrame(put_heatmap_prices,
                              index=[f"{vol:.2f}" for vol in heatmap_vol_range],
                              columns=[f"{spot:.2f}" for spot in heatmap_spot_range])
put_heatmap_df.index.name = 'Volatility (σ)'
put_heatmap_df.columns.name = 'Underlying Price (S)'


# Plot the heatmaps sequentially (one above the other)
st.subheader("Call Option Heatmap")
fig_heatmap_call, ax_heatmap_call = plt.subplots(figsize=(10, 7))
sns.heatmap(call_heatmap_df, annot=True, fmt=".2f", cmap="viridis", ax=ax_heatmap_call, cbar_kws={'label': 'Call Option Price ($)'})
ax_heatmap_call.set_title('Call Price vs. Underlying Price and Volatility')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
fig_heatmap_call.tight_layout()
st.pyplot(fig_heatmap_call)

st.subheader("Put Option Heatmap")
fig_heatmap_put, ax_heatmap_put = plt.subplots(figsize=(10, 7))
sns.heatmap(put_heatmap_df, annot=True, fmt=".2f", cmap="plasma", ax=ax_heatmap_put, cbar_kws={'label': 'Put Option Price ($)'})
ax_heatmap_put.set_title('Put Price vs. Underlying Price and Volatility')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  
fig_heatmap_put.tight_layout()
st.pyplot(fig_heatmap_put)


# --- Footer ---
st.markdown("---")
st.caption("Disclaimer: This calculator is for educational purposes only and should not be used for making actual investment decisions.")