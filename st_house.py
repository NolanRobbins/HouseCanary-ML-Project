import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your pre-trained model and scaler
with open("data/HouseCanary_model.pkl", "rb") as f:
    model = pickle.load(f)

# Ensure this is the actual RobustScaler object, not the scaled data
with open("data/robust_scaler.pkl", "rb") as f:  # Rename to the correct scaler file
    scaler = pickle.load(f)

# Load the columns from X_train_new and X_train_selected
X_train_scaled = pd.read_excel("data/X_train_scaled.xlsx")
X_train_selected = pd.read_excel("data/X_train_selected.xlsx")

# List of all 50 states (sorted to match one-hot encoding)
states = sorted(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
                 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
                 'VA', 'WA', 'WV', 'WI', 'WY'])

# --- Streamlit App ---

st.title("Build Your Dream Home Price Estimator")

with st.form("dream_home_form"):
    st.header("Enter Your Home’s Details")
    # Dropdowns for categorical values (sorted states to match encoding)
    hc_condition_class = st.selectbox("HC Condition Class", [2.0, 3.0, 4.0, 5.0, 6.0])
    state = st.selectbox("State", states)  # Now sorted
    property_type = st.selectbox("Property Type", ["SFR", "Other"])
    roof_material = st.selectbox("Roof Material", ["Shingles", "Asphalt", "Other"])
    exterior_wall_material = st.selectbox("Exterior Wall Material", 
                                         ["Siding", "Wood", "Brick", "Stucco", "Brick Veneer", "Other"])
    building_condition_code = st.selectbox("Building Condition Code", 
                                          ["Unsound", "Poor", "Fair", "Average", "Good", "Excellent", "Other"])
    bedrooms = st.selectbox("Bedrooms", ["2 or Less", "3", "4", "More than 4 Bedrooms"])
    bathrooms = st.selectbox("Bathrooms", ["1", "2", "3", "4 or More"])
    parking_garage = st.selectbox("Parking Garage", ["Street Parking", "1 Garage", "2 Garages", "More than 2 Garages"])
    parking_total = st.selectbox("Parking Total", ["Paid Parking", "1 Spot", "2 Spots", "3 Spots", "More than 3 Spots"])
    total_rooms = st.selectbox("Total Rooms", ["1-5 Rooms", "6 Rooms", "7 Rooms", "More than 7 Rooms"])
    stories = st.selectbox("Stories", ["1 Story", "2 Stories", "3 or More Stories"])
    pool = st.selectbox("Pool", [0.0, 1.0])
    
    # Sliders for continuous values
    year_built = st.slider("Year Built", min_value=1900, max_value=2023, value=2000)
    living_area = st.slider("Living Area (sq ft)", min_value=400, max_value=5500, value=1500)
    last_value_appraisal = st.slider("Last Value Appraisal Year", min_value=2017, max_value=2024, value=2020)
    
    submit_button = st.form_submit_button("What's my Price?")

if submit_button:
    # --- Preprocessing ---
    data = {
        "hc_condition_class": [hc_condition_class],
        "year_built": [year_built],
        "living_area": [living_area],
        "pool_yn": [pool],
        "value_assessed_year": [last_value_appraisal]
    }
    
    # Correct state encoding using sorted index
    state_encoded = {f"state_{i}": 1 if state == s else 0 for i, s in enumerate(states)}
    data.update(state_encoded)
    
    # Map categorical values to binned features
    binned_mappings = {
        "parking_garage_binned": parking_garage,
        "rooms_total_binned": total_rooms,
        "stories_number_binned": stories,
        "property_type_binned": property_type,
        "bedrooms_binned": bedrooms,
        "roof_type_binned": roof_material,
        "bathrooms_total_binned": bathrooms,
        "building_condition_code": building_condition_code,
        "wall_type_binned": exterior_wall_material,
        "parking_total_binned": parking_total
    }
    
    for prefix, value in binned_mappings.items():
        data[f"{prefix}_{value.replace(' ', '_')}"] = 1
    
    # Fill missing columns with 0
    for col in X_train_scaled.columns:
        if col not in data:
            data[col] = [0]
    
    input_df = pd.DataFrame(data)
    input_df = input_df.reindex(X_train_scaled.columns, axis=1, fill_value=0) 
    
    # --- Scaling ---
    # Scale all 44 features
    scaled_input = scaler.transform(input_df)
    
    # Convert to DataFrame and select the 10 features
    scaled_df = pd.DataFrame(scaled_input, columns=X_train_scaled.columns)
    input_df_selected = scaled_df[X_train_selected.columns]
    
    # Add debug checks
    print("Scaler expected columns:", scaler.feature_names_in_)
    print("Input columns:", input_df.columns.tolist())

    # Verify exact match
    assert (input_df.columns == scaler.feature_names_in_).all(), "Column order/names mismatch!"
    
    # --- Prediction ---
    log_price_pred = model.predict(input_df_selected)
    price_pred = np.exp(log_price_pred)
    
    st.success(f"Estimated Home Price: ${price_pred[0]:,.2f}")