import streamlit as st
from PIL import Image
import base64


def display_about():
    st.markdown('# Procurement.AI :arrow_forward:')
    st.markdown('<style>h1{color: #B51E60 ;}</style>', unsafe_allow_html=True)
    st.text('Created by Animesh Srivastava')
    # img = Image.open('Img2.jpg')
    # st.image(img,use_column_width=True)
    file_ = open("giphy1.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width=700>',
        unsafe_allow_html=True,
    )
    st.markdown('## About')

    st.markdown('''
    Advanced-analytics techniques use algorithms to recognize patterns in complex data sets, allowing procurement analysts to query all their data, determine the statistically significant drivers of price, and cluster the data according to those drivers. The resulting clusters represent a set of purchases without significant
    differences in cost drivers and thus reveal the real differences in vendor performance.
    
    **Procurement.AI** is an initiative by Data Science COE to create an end-to-end AI enabled product to
    simplify procurement in organizations'''
                )
    st.text("")
    st.text("")
    
    img = Image.open('Img6.png')
    st.image(img,width=700)
    st.text("")
    st.text("")
    st.markdown('## Limitations')
    st.markdown('''- Supports only CSV as input'''

                )



    st.text("")
    st.text("")
    st.markdown("## This app is developed using streamlit")
    st.markdown("""
    Streamlit’s open-source app framework is the easiest way for data scientists and machine learning engineers 
    to create beautiful, performant apps in only a few hours!  
    - Completely built in python
    - Open Source
    - Supports Ml/DL models seamlessly
    """)
    # st.text("")
    # """### gif from local file"""
    # file_ = open("vid2.gif", "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()
    #
    # st.markdown(
    #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width=700>',
    #     unsafe_allow_html=True,
    # )
    st.text("")
    st.text("")
    st.markdown("## Watch How is AI Disrupting Procurement ")
    st.text("")
    # st.video('https://www.youtube.com/watch?v=b-U9uIMFT2U&feature=youtu.be')
    st.video('https://youtu.be/tBBxn_ZHIZY',start_time=80)

def display_sidebar():
    st.sidebar.markdown('---')
    
    st.sidebar.title('Disclaimer')
    st.sidebar.info('This is a WIP product by the COE Data Science and subsequent features will be added')
    st.sidebar.title('About')
    st.sidebar.info('''This app has been developed by the [COE Data Science team](https://hcl.com)
                    using [Python](https://www.python.org/),
    [Streamlit](https://streamlit.io/), and [Plotly](https://plotly.com/python/)
                    ''')

