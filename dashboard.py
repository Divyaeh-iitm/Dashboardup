import streamlit as st
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
from scipy.stats import skew
from io import BytesIO

# -------------------------------
# Custom Rounding Function
# -------------------------------
def custom_round(x):
    try:
        val = float(x)
        # If abs(val) is less than 1 (but non-zero), round to 2 decimals
        if 0 < abs(val) < 1:
            return round(val, 2)
        else:
            return int(round(val))
    except:
        return x

# -------------------------------
# Page Configuration & Custom CSS
# -------------------------------
st.set_page_config(layout="wide")

# Updated custom CSS to force the multiselect token bubbles to appear in blue with white text.
# We're targeting any span inside the multiselect container.
st.markdown("""
    <style>
    /* Target any span within a multiselect (stMultiSelect) container */
    div[data-testid="stMultiSelect"] span {
         background-color: #1f77b4 !important;
         color: white !important;
         border-radius: 4px !important;
         padding: 2px 6px !important;
         margin: 2px !important;
         font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Choose a Plotly template and define a discrete color palette
COLOR_TEMPLATE = "plotly_white"

# ===============================
# Step 0: Basic Project Details (Sidebar Expander)
# ===============================
with st.sidebar.expander("Basic Project Details", expanded=True):
    st.header("Project Details")
    project_name = st.text_input("Project Name:")
    project_area = st.text_input("Project Area (in square feet):")
    project_location = st.text_input("Project Location:")
    work_package = st.selectbox("Work Package:", 
                                ["Shuttering", "Concreting", "Flooring", "Painting", 
                                 "False Ceiling", "CP & Sanitary Works", "Railing and Metal Works", "Windows, Doors"])
    declared_unit = st.text_input("Declared Unit (e.g., sqm):", value="sqm")

# ===============================
# Helper Functions
# ===============================
def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series >= lower) & (series <= upper)]

def rename_columns(df):
    col_mapping = {
        'embodied energy': 'embodied energy',
        'embodied carbon': 'embodied carbon'
    }
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns=col_mapping, inplace=True)
    return df

def analyze_material(df, material_name):
    """
    Analyze a material:
      - Compute skewness for EE and EC
      - Remove outliers and compute inclusive median, Q1, Q3, min, and max.
      - Generate vibrant box plots with annotations.
    """
    df = rename_columns(df)
    if 'embodied energy' not in df.columns or 'embodied carbon' not in df.columns:
        raise KeyError(f"Sheet '{material_name}' is missing required columns.")
    ee = df['embodied energy'].dropna()
    ec = df['embodied carbon'].dropna()
    ee_skew = skew(ee) if len(ee) > 0 else np.nan
    ec_skew = skew(ec) if len(ec) > 0 else np.nan
    ee_clean = remove_outliers(ee)
    ec_clean = remove_outliers(ec)
    ee_median = ee_clean.median() if not ee_clean.empty else np.nan
    ec_median = ec_clean.median() if not ec_clean.empty else np.nan
    ee_q1 = ee_clean.quantile(0.25) if not ee_clean.empty else np.nan
    ee_q3 = ee_clean.quantile(0.75) if not ee_clean.empty else np.nan
    ee_min = ee_clean.min() if not ee_clean.empty else np.nan
    ee_max = ee_clean.max() if not ee_clean.empty else np.nan
    ec_q1 = ec_clean.quantile(0.25) if not ec_clean.empty else np.nan
    ec_q3 = ec_clean.quantile(0.75) if not ec_clean.empty else np.nan
    ec_min = ec_clean.min() if not ec_clean.empty else np.nan
    ec_max = ec_clean.max() if not ec_clean.empty else np.nan

    ee_box = px.box(df, y='embodied energy', points="all", template=COLOR_TEMPLATE,
                    color_discrete_sequence=px.colors.qualitative.Dark2,
                    title=f"{material_name} – EE Distribution",
                    labels={'embodied energy': f'MJ/{declared_unit}'})
    ec_box = px.box(df, y='embodied carbon', points="all", template=COLOR_TEMPLATE,
                    color_discrete_sequence=px.colors.qualitative.Dark2,
                    title=f"{material_name} – EC Distribution",
                    labels={'embodied carbon': f'kg CO2eq./{declared_unit}'})
    ee_box.add_annotation(
        text=f"Median: {custom_round(ee_median)}<br>Q1: {custom_round(ee_q1)} & Q3: {custom_round(ee_q3)}<br>Range: [{custom_round(ee_min)}, {custom_round(ee_max)}]",
        xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False, font=dict(color="black", size=12))
    ec_box.add_annotation(
        text=f"Median: {custom_round(ec_median)}<br>Q1: {custom_round(ec_q1)} & Q3: {custom_round(ec_q3)}<br>Range: [{custom_round(ec_min)}, {custom_round(ec_max)}]",
        xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False, font=dict(color="black", size=12))
    
    stats = {
        "EE Skew": custom_round(ee_skew),
        "EE Median": custom_round(ee_median),
        "EE Q1": custom_round(ee_q1),
        "EE Q3": custom_round(ee_q3),
        "EE Range": f"[{custom_round(ee_min)}, {custom_round(ee_max)}]",
        "EC Skew": custom_round(ec_skew),
        "EC Median": custom_round(ec_median),
        "EC Q1": custom_round(ec_q1),
        "EC Q3": custom_round(ec_q3),
        "EC Range": f"[{custom_round(ec_min)}, {custom_round(ec_max)}]"
    }
    return {"stats": stats, "ee_box": ee_box, "ec_box": ec_box}

# ===============================
# Step 1: Data Upload & Material Analysis
# ===============================
st.header("Step 1: Data Upload & Material Analysis")
uploaded_file = st.file_uploader("Upload an Excel file (one sheet per material):", type=["xlsx"], key="data_upload")
if uploaded_file:
    all_data = pd.read_excel(uploaded_file, sheet_name=None)
    sheet_names = list(all_data.keys())
    st.success(f"Loaded {len(sheet_names)} materials: {', '.join(sheet_names)}")
    
    analysis_dict = {}
    summary_list = []
    for material in sheet_names:
        try:
            analysis = analyze_material(all_data[material], material)
            analysis_dict[material] = analysis
            s = analysis["stats"]
            summary_list.append({
                "Material": material,
                "EE Skew": s["EE Skew"],
                "EE Median": s["EE Median"],
                "EE Q1": s["EE Q1"],
                "EE Q3": s["EE Q3"],
                "EE Range": s["EE Range"],
                "EC Skew": s["EC Skew"],
                "EC Median": s["EC Median"],
                "EC Q1": s["EC Q1"],
                "EC Q3": s["EC Q3"],
                "EC Range": s["EC Range"]
            })
        except Exception as e:
            st.warning(f"Skipping {material}: {e}")
    summary_df = pd.DataFrame(summary_list)
    st.subheader("Material Analysis Summary")
    st.dataframe(summary_df)
    
    with st.expander("Show/Hide Material Box Plots"):
        st.subheader("Material Box Plots")
        cols = st.columns(3)
        for idx, material in enumerate(sheet_names):
            with cols[idx % 3]:
                st.plotly_chart(analysis_dict[material]["ee_box"], use_container_width=True)
                st.plotly_chart(analysis_dict[material]["ec_box"], use_container_width=True)

# ===============================
# Step 2: Mapping of Secondary Materials
# ===============================
st.header("Step 2: Mapping of Secondary Materials")
if uploaded_file:
    st.subheader("Select Primary Materials for System Mapping")
    primary_materials = st.multiselect("Primary Materials:", options=sheet_names, key="primary_select")
    
    st.subheader("Define Secondary Material Categories (SM1 to SM5)")
    secondary_materials = {}
    selected_scat = {}
    for i in range(1, 6):
        sm_key = f"SM{i}"
        sm_purpose = st.text_input(f"Purpose for {sm_key} (e.g., fixing, finishing):", key=f"purpose_{sm_key}")
        if sm_purpose.lower() == "n/a":
            break
        prev_selected = set()
        for j in range(1, i):
            prev_selected = prev_selected.union(set(selected_scat.get(f"SM{j}", [])))
        available = [m for m in sheet_names if (m not in primary_materials and m not in prev_selected)]
        selected = st.multiselect(f"Select materials for {sm_key} (Purpose: {sm_purpose}):", options=available, key=f"list_{sm_key}")
        secondary_materials[sm_key] = {"purpose": sm_purpose, "materials": selected}
        selected_scat[sm_key] = selected
    
    st.subheader("Map Secondary Materials to Primary Materials with Dependency Options")
    mappings = {}
    for primary in primary_materials:
        st.markdown(f"**Mapping for {primary}:**")
        mappings[primary] = {}
        for sm_key, sm_data in secondary_materials.items():
            st.markdown(f"*For {sm_key} ({sm_data['purpose']}):*")
            selected_items = st.multiselect(f"Select items from {sm_key} for {primary}:", options=sm_data["materials"], key=f"map_{primary}_{sm_key}")
            mapping_entries = []
            if sm_key == "SM2":
                for item in selected_items:
                    dep_required = st.checkbox(f"Does '{item}' require dependency?", key=f"dep_check_{primary}_{sm_key}_{item}")
                    if dep_required and "SM1" in secondary_materials and secondary_materials["SM1"]["materials"]:
                        dep = st.selectbox(f"Select dependency for '{item}' (from SM1):", options=secondary_materials["SM1"]["materials"], key=f"dep_{primary}_{sm_key}_{item}")
                        mapping_entries.append({"material": item, "dependency": dep})
                    else:
                        mapping_entries.append({"material": item, "dependency": None})
            else:
                for item in selected_items:
                    mapping_entries.append({"material": item, "dependency": None})
            mappings[primary][sm_key] = mapping_entries

# ===============================
# Step 3: System Generation and Analysis
# ===============================
st.header("Step 3: System Generation and Analysis")
if uploaded_file and primary_materials and secondary_materials and mappings:
    unique_primary = list(primary_materials)
    colors = px.colors.qualitative.Dark2
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(unique_primary)}
    
    system_results = []
    system_counter = 1
    for primary in primary_materials:
        mapping_lists = []
        for sm_key in secondary_materials.keys():
            if sm_key == "SM1":
                entries = mappings[primary].get("SM1", [])
                indep = [entry["material"] for entry in entries]
                dep_from_sm2 = [entry["dependency"] for entry in mappings[primary].get("SM2", []) if entry["dependency"]]
                effective_sm1 = list(set(indep).union(set(dep_from_sm2)))
                effective_entries = [[x] for x in effective_sm1] if effective_sm1 else [[None]]
                mapping_lists.append((sm_key, effective_entries))
            elif sm_key == "SM2":
                entries = mappings[primary].get("SM2", [])
                effective_entries = [[entry["material"]] for entry in entries] if entries else [[None]]
                mapping_lists.append((sm_key, effective_entries))
            else:
                entries = mappings[primary].get(sm_key, [])
                effective_entries = [[entry["material"]] for entry in entries] if entries else [[None]]
                mapping_lists.append((sm_key, effective_entries))
        if mapping_lists:
            categories = [t[0] for t in mapping_lists]
            options_per_cat = [t[1] for t in mapping_lists]
            for combo in itertools.product(*options_per_cat):
                total_ee = analysis_dict[primary]["stats"]["EE Median"]
                total_ec = analysis_dict[primary]["stats"]["EC Median"]
                effective_map = {}
                for idx, sm_cat in enumerate(categories):
                    eff_list = combo[idx]
                    vals = []
                    for val in eff_list:
                        if val is not None:
                            total_ee += analysis_dict[val]["stats"]["EE Median"]
                            total_ec += analysis_dict[val]["stats"]["EC Median"]
                            vals.append(val)
                    effective_map[sm_cat] = ", ".join(vals) if vals else "N/A"
                system_results.append({
                    "System": f"S{system_counter}",
                    "Primary": primary,
                    **effective_map,
                    "EE (MJ/"+declared_unit+")": custom_round(total_ee),
                    "EC (kg CO2eq./"+declared_unit+")": custom_round(total_ec)
                })
                system_counter += 1
        else:
            system_results.append({
                "System": f"S{system_counter}",
                "Primary": primary,
                "EE (MJ/"+declared_unit+")": custom_round(analysis_dict[primary]["stats"]["EE Median"]),
                "EC (kg CO2eq./"+declared_unit+")": custom_round(analysis_dict[primary]["stats"]["EC Median"])
            })
            system_counter += 1
    df_systems = pd.DataFrame(system_results)
    ee_sorted = df_systems.sort_values("EE (MJ/"+declared_unit+")")
    ec_sorted = df_systems.sort_values("EC (kg CO2eq./"+declared_unit+")")
    
    st.subheader("System Analysis Results")
    st.write("### Embodied Energy (EE):")
    st.dataframe(ee_sorted)
    st.write("### Embodied Carbon (EC):")
    st.dataframe(ec_sorted)
    
    st.write("### EE Comparison")
    fig_ee = px.bar(ee_sorted, x="System", y="EE (MJ/"+declared_unit+")", title="Systems - Embodied Energy", color="Primary", 
                      color_discrete_map=color_map, template=COLOR_TEMPLATE)
    st.plotly_chart(fig_ee, use_container_width=True)
    
    st.write("### EC Comparison")
    fig_ec = px.bar(ec_sorted, x="System", y="EC (kg CO2eq./"+declared_unit+")", title="Systems - Embodied Carbon", color="Primary",
                      color_discrete_map=color_map, template=COLOR_TEMPLATE)
    st.plotly_chart(fig_ec, use_container_width=True)
    
    # ===============================
    # Step 4: Base Case Comparison
    # ===============================
    st.header("Step 4: Base Case Comparison")
    base_case = st.selectbox("Select a base system for comparison:", options=df_systems["System"].unique())
    if base_case:
        base_row = df_systems[df_systems["System"] == base_case].iloc[0]
        base_ee = base_row[f"EE (MJ/{declared_unit})"]
        base_ec = base_row[f"EC (kg CO2eq./{declared_unit})"]
        df_systems["EE Change (%)"] = df_systems[f"EE (MJ/{declared_unit})"].apply(lambda x: custom_round(((x - base_ee) / base_ee) * 100) if base_ee != 0 else 0)
        df_systems["EC Change (%)"] = df_systems[f"EC (kg CO2eq./{declared_unit})"].apply(lambda x: custom_round(((x - base_ec) / base_ec) * 100) if base_ec != 0 else 0)
        st.subheader("Base Case Comparison Table")
        st.dataframe(df_systems.sort_values(f"EE (MJ/{declared_unit})"))
        st.write("### EE Change (%) Comparison")
        fig_change_ee = px.bar(df_systems, x="System", y="EE Change (%)", title="Percentage Change in EE Relative to Base Case", color="Primary", color_discrete_map=color_map, template=COLOR_TEMPLATE)
        st.plotly_chart(fig_change_ee, use_container_width=True)
        st.write("### EC Change (%) Comparison")
        fig_change_ec = px.bar(df_systems, x="System", y="EC Change (%)", title="Percentage Change in EC Relative to Base Case", color="Primary", color_discrete_map=color_map, template=COLOR_TEMPLATE)
        st.plotly_chart(fig_change_ec, use_container_width=True)
    
