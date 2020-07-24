import streamlit as st
from PIL import Image
import base64
#filename = st.file_picker("Pick a file", folder="my_folder", type=("png", "jpg"))

#EDEEF2
def display_about():
    st.markdown('<style>body{background-color:#EDEEF2; color: Black}</style>',unsafe_allow_html=True)
    # st.title("Procurement.AI :arrow_forward:")
    #
    # st.markdown("<h1 style='text-align: center;font-size:60px;color:#B51E60;'>Procurement.AI</h1>", unsafe_allow_html=True)
    st.markdown('<style>h1{color: #B51E60 ;}</style>', unsafe_allow_html=True)
    # st.text('Created by Animesh Srivastava as a part of COE')
    # img = Image.open('Img2.jpg')
    # st.image(img,use_column_width=True)
    file_ = open("brain3.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width=700>',
        unsafe_allow_html=True,
    )
    st.markdown('# About')

    htm_tag= ('''
    <div style='text-align: justify;'> Advanced-analytics techniques use algorithms to recognize patterns in complex data sets, 
    allowing procurement analysts to query all their data, determine the statistically significant 
    drivers of price, and cluster the data according to those drivers. The resulting clusters represent 
    a set of purchases without significant differences in cost drivers and thus reveal the real differences in 
    vendor performance.
    
    **Procurement.AI** is an initiative by Data Science COE to create an end-to-end AI enabled product to
    simplify procurement in organizations </div>'''
                )
    st.markdown(htm_tag,unsafe_allow_html=True)
    st.text("")
    st.text("")
    
    img = Image.open('Img6_1.png')
    st.image(img,width=700)
    st.text("")
    st.text("")
    st.markdown('# Limitations')
    st.markdown('''
                    - Supports only CSV as input
                    -  Currently work in progress for the entire suite of use cases
                    '''

                )



    st.text("")
    st.text("")
    st.markdown("# This app is developed using streamlit")
    html_tag=(
    '''
    <div style='text-align: justify;'> Streamlit’s open-source app framework is the easiest way for data scientists and machine learning engineers 
    to create beautiful, performant apps in only a few hours!  
    - Completely built in python
    - Open Source
    - Supports Ml/DL models seamlessly </div
    ''')
    st.markdown(html_tag,unsafe_allow_html=True)
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
    st.markdown("# Watch How is AI Disrupting Procurement ")
    st.text("")
    # st.video('https://www.youtube.com/watch?v=b-U9uIMFT2U&feature=youtu.be')
    st.video('https://youtu.be/tBBxn_ZHIZY',start_time=80)

def display_sidebar():
    st.sidebar.markdown('---')
    
    st.sidebar.title('Disclaimer')
    st.sidebar.info('This is a WIP product by the COE Data Science and subsequent features will be added')
    st.sidebar.title('About')
    st.sidebar.info('''This app has been developed by [Animesh Srivastava](https://www.linkedin.com/in/animesh-srivastava-87968880/)
                    using [Python](https://www.python.org/),
    [Streamlit](https://streamlit.io/), and [Plotly](https://plotly.com/python/)
                    ''')

