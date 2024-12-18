import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

# Function to load data in chunks
@st.cache_data
def load_data_in_chunks(filepath, chunksize=1000):
    chunk_list = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk_list.append(chunk)
    return pd.concat(chunk_list)

# Load a manageable chunk of the dataset
material_df = load_data_in_chunks('clean_filtered_materialsdata.csv')

# Define the feature columns and target column
X = material_df.drop(columns=['efermi', 'band_gap', 'composition', 'is_metal'])
y = material_df['band_gap']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Extra Trees model with the best parameters
extra_trees_model = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42)
extra_trees_model.fit(X_scaled, y)

# Streamlit app
st.title('Material Band Gap Prediction 🔮')
st.write("Use this app to predict the band gap of materials based on their properties. 🧪")

# Sidebar for input fields
st.sidebar.header('Input Features 📝')
if 'input_data' not in st.session_state:
    st.session_state.input_data = {column: 0.0 for column in X.columns}

with st.sidebar.form(key='input_form'):
    for column in X.columns:
        st.session_state.input_data[column] = st.number_input(f'Enter {column}', value=st.session_state.input_data[column], format="%.8f")
    
    # Add a submit button to the form
    submit_button = st.form_submit_button(label='Run 🚀')

# Center section with instructions
st.write("## How to Use the App 🛠️")
st.write("""
1. **Enter the values for each feature** in the sidebar. The features include:
    - **is_gap_direct**: Whether the band gap is direct (0 or 1).
    - **nsites**: Number of sites in the material.
    - **nelements**: Number of different elements in the material.
    - **volume**: Volume of the material.
    - **density**: Density of the material.
    - **density_atomic**: Atomic density of the material.
    - **uncorrected_energy_per_atom**: Uncorrected energy per atom.
    - **energy_per_atom**: Energy per atom.
    - **formation_energy_per_atom**: Formation energy per atom.
    - **energy_above_hull**: Energy above the hull.
    - **is_stable**: Stability of the material (0 or 1).
    - **is_magnetic**: Whether the material is magnetic (0 or 1).
    - **ordering**: Ordering of the material.
    - **total_magnetization**: Total magnetization of the material.
    - **total_magnetization_normalized_vol**: Total magnetization normalized by volume.
    - **total_magnetization_normalized_formula_units**: Total magnetization normalized by formula units.
    - **num_magnetic_sites**: Number of magnetic sites.
    - **num_unique_magnetic_sites**: Number of unique magnetic sites.
    - **theoretical**: Whether the data is theoretical (0 or 1).

2. **Click the "Run" button** to predict the band gap.

3. **View the results**: The predicted band gap will be displayed below.
""")

# Run the prediction only when the form is submitted
if submit_button:
    with st.spinner('Predicting... 🔄'):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([st.session_state.input_data])

        # Standardize the input data
        input_scaled = scaler.transform(input_df)

        # Check if the band gap already exists in the dataset
        existing_band_gap = material_df.loc[
            (material_df[X.columns] == input_df.values).all(axis=1), 'band_gap'
        ]

        if not existing_band_gap.empty:
            st.success(f'The band gap for the given composition already exists: {existing_band_gap.values[0]} 🎉')
        else:
            # Predict the band gap using the Extra Trees model
            predicted_band_gap = extra_trees_model.predict(input_scaled)
            st.success(f'Predicted Band Gap: {predicted_band_gap[0]} ⚡')

    # Display additional information
    st.write("### Model Information 📊")
    st.write("This prediction is made using an Extra Trees Regressor model trained on the provided dataset. The model uses the following features to predict the band gap:")
    st.write(", ".join(X.columns))