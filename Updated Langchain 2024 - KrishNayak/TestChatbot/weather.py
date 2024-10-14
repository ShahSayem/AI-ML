import pandas as pd

# Define data for the weather of a month in a fictional location called NWC (Northwest City)
data = {
    "Date": pd.date_range(start="2024-09-01", end="2024-09-30"),
    "Temperature (°C)": [25, 24, 26, 27, 26, 25, 24, 26, 28, 27, 26, 25, 24, 23, 22, 24, 26, 25, 27, 28, 26, 24, 23, 22, 24, 26, 28, 29, 30, 27],
    "Weather Condition": ["Sunny", "Cloudy", "Sunny", "Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Sunny", "Sunny", "Sunny", "Cloudy", "Rainy", "Sunny", "Sunny", 
                          "Rainy", "Sunny", "Sunny", "Cloudy", "Sunny", "Partly Cloudy", "Sunny", "Rainy", "Sunny", "Sunny", "Cloudy", "Rainy", "Sunny", 
                          "Sunny", "Partly Cloudy", "Sunny"],
    "Humidity (%)": [60, 65, 62, 61, 63, 68, 70, 62, 59, 60, 66, 72, 61, 60, 75, 62, 59, 64, 60, 63, 62, 71, 60, 59, 67, 73, 61, 58, 57, 62],
    "Wind Speed (km/h)": [12, 14, 10, 9, 15, 16, 17, 10, 8, 9, 12, 18, 10, 9, 16, 11, 8, 13, 10, 14, 9, 17, 10, 9, 14, 18, 10, 8, 7, 9]
}

# Create DataFrame
weather_df = pd.DataFrame(data)

# Save as Excel file
weather_df.to_excel("nwc_sep24.xlsx", index=False)
