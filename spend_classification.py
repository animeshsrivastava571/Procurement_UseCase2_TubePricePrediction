import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# Plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode()

from PIL import Image

def spend_classify():
    st.markdown("# Spend Classification :bulb:")
    st.markdown('''Spend analysis is a critical first-step to establishing a truly effective procurement 
                organization and presents a major challenge to organizations with limited time and resources. 
                However, artificial intelligence, specifically machine learning, can increase speed to spend 
                analysis results and overall accuracy while reducing the amount of manual input.''')
    img = Image.open('tb_img3.png')
    st.image(img,width=900)

    st.markdown("## Look at the Sample Data")
    df1 = pd.read_csv('D:/HCL_office/Use_Case3_SpendClassification/purchase-order-quantity-price-detail-for-commodity-goods-procurements-1.csv',nrows=10)
    st.table(df1)

    st.markdown('''## Challenges''')
    st.markdown('''
     - Absence of Labelled Data
    - Choice of Classes and the level of intricacy that we can achieve
    ''')
    return 0
# spend_classify()
