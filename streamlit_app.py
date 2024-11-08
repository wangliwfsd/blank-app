import streamlit as st
# import numpy as np
import pandas as pd
import pickle
# from sklearn.ensemble import RandomForestClassifier

# 加载训练好的随机森林模型和列信息
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('columns.pkl', 'rb') as file:
    columns = pickle.load(file)

# 定义输入特征的名称和类别特征的选项
features = [
    'Commissioning year', 'Capacity 10^3 (m3/d)', 'Feed TDS (mg/L)', 
    'Product TDS (mg/L)', 'Remin', 'Temp. (℃)', 'Pre-pretreatment', 
    'Pretreatment', 'Overall configuration', 'SWRO Configuration', 
    'SWRO Recovery (%)', 'BWRO Configuration', 'BWRO Recovery (%)', 'ERD type'
]

numerical_features = {
    'Commissioning year': 2011,
    'Capacity 10^3 (m3/d)': 200.0,
    'Feed TDS (mg/L)': 35900,
    'Product TDS (mg/L)': 0.0,  # 默认值设为0.0，因其为NaN
    'Temp. (℃)': 21.0,
    'SWRO Recovery (%)': 45.0,
    'BWRO Recovery (%)': 0.0  # 默认值设为0.0，因其为NaN
}

categorical_features = {
    'Remin': ['b', 'a'],
    'Pre-pretreatment': ['DAF'],
    'Pretreatment': ['DMF', 'UF', 'UF/DMF', 'MF'],
    'Overall configuration': ['Single pass', 'Partial two pass', 'SPSP', 'Full two pass',
                              'Advanced SPSP', 'Two pass', 'Full triple pass',
                              'Full/Partial two pass'],
    'SWRO Configuration': ['Single stage', 'Two stage'],
    'BWRO Configuration': ['Two stage', 'Cascade', 'Three stage', 'Two pass', 'PCP'],
    'ERD type': ['PX', 'FT', 'PT', 'DWEER']
}
default_values = {
    'Commissioning year': 2011,
    'Capacity 10^3 (m3/d)': 200.0,
    'Feed TDS (mg/L)': 35900,
    'Product TDS (mg/L)': 0.0,  # 默认值设为0.0，因其为NaN
    'Remin': 'b',  # 默认值设为'b'，因其为NaN
    'Temp. (℃)': 21.0,
    'Pre-pretreatment': 'DAF',  # 默认值设为'DAF'，因其为NaN
    'Pretreatment': 'DMF',
    'Overall configuration': 'Single pass',
    'SWRO Configuration': 'Single stage',
    'SWRO Recovery (%)': 45.0,
    'BWRO Configuration': 'Two stage',  # 默认值设为'Two stage'，因其为NaN
    'BWRO Recovery (%)': 0.0,  # 默认值设为0.0，因其为NaN
    'ERD type': 'PX'
}

# 创建输入界面
st.title('SEC Predict')
st.write('Please enter the feature values and click the Predict button to get the prediction results.')

input_values = []
col1, col2 = st.columns(2)
with col1:
    # 输入数值特征
    for feature_name in numerical_features:
        input_value = st.number_input(f'{feature_name}', value=default_values[feature_name])
        input_values.append(input_value)

# 输入类别特征
with col2:
    categorical_input_values = {}
    for feature_name, options in categorical_features.items():
        default_value = default_values.get(feature_name, options[0])
        selected_option = st.selectbox(feature_name, options, index=options.index(default_value))
        categorical_input_values[feature_name] = selected_option

# 将数值特征和类别特征组合成一个DataFrame
input_data = {**{feature_name: [input_values[i]] for i, feature_name in enumerate(numerical_features)},
              **{feature_name: [categorical_input_values[feature_name]] for feature_name in categorical_input_values}}

input_df = pd.DataFrame(input_data)

# 对类别特征进行编码
input_onehot = pd.get_dummies(input_df, drop_first=True).reindex(columns=columns, fill_value=0)

# 进行预测
if st.button('Predict'):
    prediction = model.predict(input_onehot)
    # st.write(f'Predicted SEC: {prediction[0]}')
    html_str = f"""
    <style>
    p.a {{
    font: bold {25}px Courier;
    }}
    </style>
    <p class="a">Predicted SEC: {prediction[0]}</p>
    """

    st.markdown(html_str, unsafe_allow_html=True)
