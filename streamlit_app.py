import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Streamlit page setup
st.set_page_config(layout="wide")
st.title("🏡 UK House Prices and Earnings Analysis")

# Sidebar filters and setup
st.sidebar.header("🔍 Filter Data")

# Sample data loading (for demonstration purpose without file upload)
df_1a = pd.read_excel("house-price-to-residence-based-earnings.xlsx", sheet_name="1a", skiprows=1, engine='openpyxl')
df_1b = pd.read_excel("house-price-to-residence-based-earnings.xlsx", sheet_name="1b", skiprows=1, engine='openpyxl')

# Data preparation
df_1a.columns = df_1a.columns.str.replace('Year ending Sep ', '', regex=False)
df_1a_melted = df_1a.melt(id_vars=['Code', 'Name'], var_name='Year', value_name='HousePrice')
df_1b_melted = df_1b.melt(id_vars=['Code', 'Name'], var_name='Year', value_name='GrossIncome')
df_1b_melted['GrossIncome'] = df_1b_melted['GrossIncome'].astype(float)
df = df_1a_melted.merge(df_1b_melted, how='inner', on=['Code', 'Name', 'Year'])

# Convert Year to int
df['Year'] = df['Year'].astype(int)
df = df.sort_values(by=['Name', 'Year'])
df['HousePrice_Pct_Change'] = round(df.groupby('Name')['HousePrice'].pct_change() * 100, 2)

regions = st.sidebar.multiselect("Select Region(s):", options=df['Name'].unique(), default='London')
year_range = st.sidebar.slider("Select Year Range:", int(df['Year'].min()), int(df['Year'].max()), (2002, 2024))

df_filtered = df[(df['Name'].isin(regions)) & (df['Year'].between(year_range[0], year_range[1]))]

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📈 Trends", "📊 Distributions", "📉 Relationships"])

with tab1:
    st.subheader("House Prices by Region Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_filtered, x='Year', y='HousePrice', hue='Name', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("% Change in House Prices by Region")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_filtered, x='Year', y='HousePrice_Pct_Change', hue='Name', ax=ax)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    max_change_row = df[df['HousePrice_Pct_Change'] == df['HousePrice_Pct_Change'].max()]
    st.markdown(f"**Highest % change:** {max_change_row.iloc[0]['Name']} in {int(max_change_row.iloc[0]['Year'])} with {max_change_row.iloc[0]['HousePrice_Pct_Change']}%")

with tab2:
    st.subheader("House Price Distribution by Region")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_filtered, x='Name', y='HousePrice', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("House Price Histogram")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=df_filtered, x='HousePrice', hue='Name', bins=30, multiple='stack', ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("House Prices vs Gross Income")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=df_filtered, x='GrossIncome', y='HousePrice', hue='Name', ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    corr_matrix = df_filtered[['HousePrice', 'GrossIncome']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# Summary Statistics
st.sidebar.header("📌 Summary Statistics")
st.sidebar.metric("Average House Price", f"£{df_filtered['HousePrice'].mean():,.0f}")
st.sidebar.metric("Average Earnings", f"£{df_filtered['GrossIncome'].mean():,.0f}")
st.sidebar.metric("Mode House Price", f"£{df_filtered['HousePrice'].mode().iloc[0]:,.0f}")
st.sidebar.metric("Std Dev House Price", f"£{df_filtered['HousePrice'].std():,.0f}")