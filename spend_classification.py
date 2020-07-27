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
import pydeck as pdk
from wordcloud import WordCloud


from PIL import Image

def spend_classify():
    st.markdown("# Spend Classification :bulb:")
    st.markdown('''Spend analysis is a critical first-step to establishing a truly effective procurement 
                organization and presents a major challenge to organizations with limited time and resources. 
                However, artificial intelligence, specifically machine learning, can increase speed to spend 
                analysis results and overall accuracy while reducing the amount of manual input.''')
    img = Image.open('tb_img3_transparent.png')
    st.image(img,width=900)

    st.markdown("## Sample Data")
    df1 = pd.read_csv('D:/HCL_office/Use_Case3_SpendClassification/purchase-order-quantity-price-detail-for-commodity-goods-procurements-1.csv',nrows=10)
    st.table(df1)

    st.markdown('''## Challenges''')
    st.markdown('''
     - Absence of Labelled Data
    - Choice of Classes and the level of intricacy that we can achieve
    ''')
    df = pd.read_csv('cleaned_annotation.csv',index_col=0)
    st.markdown("## WordCloud of the Text Data")
    st.text("")

    text = ''.join(df.combined_text)
    wordcloud = WordCloud(width=1600, height=800).generate(text)
    # Open a plot of the generated image.

    plt.figure( figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(transparent=True)
    st.text("")

    st.markdown("## NIGP Codes")
    html_tag=(
    '''
    <div style='text-align: justify;'> 
    The NIGP Commodity/Services Code is an acronym for the National Institute of Governmental Purchasings' 
    Commodity/Services Code. The NIGP Code is a coding taxonomy used primarily to classify products and services 
    procured by state and local governments in North America.
    The classification system was developed in the mid 1980s as a result of efforts by public 
    procurement officials in Texas, Oklahoma, Florida, Illinois and other states, cities and counties to 
    provide a mechanism to classify the products and services that used in public procurement. 
    Led by Homer Forrestor, the Director of General Services in Texas, the group produced the initial codeset in 
    1983.</div
    ''')
    st.markdown(html_tag,unsafe_allow_html=True)
    nigp_codes = pd.read_csv('NIGP_Codes.csv').iloc[:10,:]
    st.text("")
    st.table(nigp_codes)



    st.markdown("## Our Data Set in Predominantly from Texas, US")
    lat =31.9686
    lng=-99.9098

    view_state = pdk.ViewState(latitude=lat, longitude=lng,
                                   zoom=11, pitch=45);
    deck = pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',
            initial_view_state=view_state);
    st.pydeck_chart(deck);

    df_final_cat_count = pd.read_csv('Category_Number.csv',index_col=0)
    df_final_cat_count=df_final_cat_count.iloc[:10,:]
    parties = list(df_final_cat_count.Category)
    seats = list(df_final_cat_count.COMMODITY)
    percent = list(df_final_cat_count.percent)
    data1 = {
                "values": seats,
                "labels": parties,

                "name": "Commodity",
                "hoverinfo":"label+percent+name",
                "hole": .4,
                "type": "pie"
            }

    data = [data1]
    st.text("")
    st.markdown("## Commodity Classification")
    layout = dict(
                          xaxis= dict(title= 'Date',ticklen= 5,zeroline= False,showgrid=False),
                          yaxis = dict(title= 'Warehouse Orders',ticklen=5,zeroline=False,showgrid=False),
                          width=900,
                          height=500,
                          hoverlabel=dict(
                                        bgcolor="white",
                                        font_size=10,
                                        font_family="Rockwell"

        ),
                          paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                  legend=dict(

                                                    y=0.5,

                                                    x=3
                                            ),

                         )
    fig = go.Figure(data = data, layout = layout)
    st.plotly_chart(fig)

    # st.markdown("## Labelling Strategy 1: TF-IDF + K-means Clustering")
    # st.text("")
    # img = Image.open('Cluster1.png')
    # st.image(img,width=900)
    # st.markdown('''### From the various metrics''')
    # st.markdown('''
    # - Inter-intra cluster distance ratio
    # - Calisnki-Harbasz Index
    # - Silhouette Score
    #     ''')
    # st.markdown('We choose **5** as the number of clusters')
    #
    # st.markdown('''### Let's have a look at the clusters''')
    # img = Image.open('Cluster3.png')
    # st.image(img,width=900)
    return 0
# spend_classify()
