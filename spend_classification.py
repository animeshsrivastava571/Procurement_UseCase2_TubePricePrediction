import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# Plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode()
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud


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
    df = pd.read_csv('cleaned_annotation.csv',index_col=0)
    st.markdown("## Look at the WordCloud of the Text Data")
    st.text("")

    text = ''.join(df.combined_text)
    wordcloud = WordCloud(width=1600, height=800).generate(text)
    # Open a plot of the generated image.

    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()
    st.text("")

    st.markdown("## Labelling Strategy 1: K-means Clustering")
    st.text("")
    img = Image.open('Ani.png')
    st.image(img,width=900)
    st.markdown('''### From the various metrics''')
    st.markdown('''
    - Inter-intra cluster distance ratio
    - Calisnki-Harbasz Index
    - Slihouette Score
        ''')
    st.markdown('We choose **7** as the number of clusters')

    st.markdown('''### Let's have a look at the clusters''')
    img = Image.open('Ani2.png')
    st.image(img,width=900)
    return 0
# spend_classify()
