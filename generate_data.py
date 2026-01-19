import pandas as pd
import numpy as np

# Create a synthetic dataset for house prices
np.random.seed(42)
n_samples = 100

gr_liv_area = np.random.randint(500, 5000, n_samples)
bedroom_abv_gr = np.random.randint(1, 6, n_samples)
full_bath = np.random.randint(1, 4, n_samples)

# Price = 100 * area + 20000 * bedrooms + 10000 * bathrooms + noise
sale_price = (120 * gr_liv_area) + (25000 * bedroom_abv_gr) + (15000 * full_bath) + np.random.randint(5000, 20000, n_samples)

df = pd.DataFrame({
    'GrLivArea': gr_liv_area,
    'BedroomAbvGr': bedroom_abv_gr,
    'FullBath': full_bath,
    'SalePrice': sale_price
})

df.to_csv('cleaned_house_prices.csv', index=False)
print("Synthetic dataset 'cleaned_house_prices.csv' created.")
