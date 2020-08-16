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
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import roc_auc_score
import eli5
from PIL import Image
from streamlit import components
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


# @st.cache(allow_output_mutation=True)
def commodity_classify():
    st.markdown("# Commodity Classification :bulb:")
    st.markdown('''Commodity Classification is a critical first-step to establishing a truly effective procurement 
                organization and presents a major challenge to organizations with limited time and resources. 
                However, artificial intelligence, specifically machine learning, can increase speed to spend 
                analysis results and overall accuracy while reducing the amount of manual input.''')
    img = Image.open('tb_img3_transparent.png')
    st.image(img,width=900)

    cat_level0 =['EDA','ML Modelling','Dashboard']
    a=st.selectbox('Select your choice of view', cat_level0,index=0)
    if a== "Dashboard":
        st.markdown("## Power BI Dashboard highlighting the KPIs")
        st.text(" ")
        st.markdown('<style>body{background-color:white;}</style>',unsafe_allow_html=True)
        st.markdown("""
        <iframe width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=86f26045-38fc-4c60-a993-87c0610b0b07&autoAuth=true&ctid=189de737-c93a-4f5a-8b68-6f4ca9941912&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLXNvdXRoLWVhc3QtYXNpYS1yZWRpcmVjdC5hbmFseXNpcy53aW5kb3dzLm5ldC8ifQ%3D%3D"></iframe>
        """, unsafe_allow_html=True)

    if a=="EDA":

        st.markdown("## Sample Data")
        @st.cache(allow_output_mutation=True)
        def load_data():
            data = pd.read_csv('D:\\HCL_office\\Use_Case3_SpendClassification\\Final_Labelled_CatGrouping_Cleaned.csv',index_col=0)
            data = data.drop('content',axis=1)

            return data
        data = load_data()
        st.table(data.iloc[:5,:])
        st.text("")

        st.markdown("## Problem statement")
        st.markdown(" Using the commodity description text, predict the category to which the corresponding commodity/spend belongs to")

        st.markdown("## WordCloud of the Text Data")
        st.text("")
        st.image("wc.PNG",use_column_width=True)

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
        nigp_codes = pd.read_csv('NIGP_Codes.csv').iloc[25:30,:]
        cat_new =["Construction","Industrial Machinery","Industrial Machinery","Others","Industrial Machinery"]
        nigp_codes['New_Category'] = cat_new
        st.text("")
        st.table(nigp_codes)

        st.markdown("## Our Data Set in Predominantly from Texas, US")
        lat =31.9686
        lng=-99.9098

        view_state = pdk.ViewState(latitude=lat, longitude=lng,
                                       zoom=5.75, pitch=45);
        deck = pdk.Deck(map_style='mapbox://styles/mapbox/dark-v9',
                initial_view_state=view_state);
        st.pydeck_chart(deck);

        st.text(" ")
        st.markdown("## Original Top 15 Class Total Amount Distribution: 73% of the total data")
        df=data.groupby(['Category'])['ITM_TOT_AM'].sum().reset_index().sort_values(by='ITM_TOT_AM',ascending=False)
        df['percent'] = df['ITM_TOT_AM'].apply(lambda x:round( (x/ df['ITM_TOT_AM'].sum())*100),2)
        df_half=df.iloc[:15,:]
        parties = list(df_half.Category)
        seats = list(df_half.ITM_TOT_AM)
        percent = list(df_half.percent)
        data1 = {
                    "values": seats,
                    "labels": parties,

                    "name": "Commodity",
                    "hoverinfo":"label+value+name",
                    "hole": .4,
                    "type": "pie",

                }

        data2 = [data1]

        layout = dict(
                              xaxis= dict(title= 'Classes',zeroline= False,showgrid=False),
                              yaxis = dict(title= 'Amount',zeroline=False,showgrid=False),
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
                x=-2,
                y=0.7,
                traceorder='normal',
                font=dict(
                    size=10,),
            ),

                             )
        fig = go.Figure(data = data2, layout = layout)
        st.plotly_chart(fig)

        st.markdown("## Modified Category Frequency Distribution, 10 categories")

        st.text(" ")
        state_counts = Counter(data['Category_Grouping'])
        df_state = pd.DataFrame.from_dict(state_counts, orient='index')
        langs = list(df_state.index)
        students = list(df_state[0])
        data2 = [go.Bar(
           x = langs,
           y = students,
            marker=dict(color='green')
        )]
        layout = dict(
                          xaxis= dict(ticklen= 5,zeroline= False,showgrid=False,title='Categories'),
                          yaxis = dict(title= 'Frequency',ticklen=5,zeroline=False,showgrid=False),
                          width=900,
                        height=700,
                       paper_bgcolor='rgba(0,0,0,0)',
              plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(data=data2,layout=layout)

        st.plotly_chart(fig)

        st.markdown("## Modified Category Amount Distribution, 10 categories")
        parties = list(data.groupby('Category_Grouping')['ITM_TOT_AM'].sum().reset_index()['Category_Grouping'])
        seats = list(data.groupby('Category_Grouping')['ITM_TOT_AM'].sum().reset_index()['ITM_TOT_AM'])
        percent = list(df_half.percent)
        data1 = {
                    "values": seats,
                    "labels": parties,

                    "name": "Commodity",
                    "hoverinfo":"label+value+name",
                    "hole": .4,
                    "type": "pie"
                }

        data2 = [data1]

        layout = dict(
                              xaxis= dict(title= 'Date',zeroline= False,showgrid=False),
                              yaxis = dict(title= 'Warehouse Orders',zeroline=False,showgrid=False),
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
                x=0,
                y=0,
                traceorder='normal',
                font=dict(
                    size=12,),
            ),

                             )
        fig = go.Figure(data = data2, layout = layout)
        st.plotly_chart(fig)


        st.markdown("## Top Vendors per Category: Total Amount")
        def transformer(df):
            df_vendor=df.groupby(['Category_Grouping','VENDOR_CODE'])['ITM_TOT_AM'].sum().reset_index().sort_values(by='ITM_TOT_AM',ascending=False)
            df_vendor['percent'] = df_vendor['ITM_TOT_AM'].apply(lambda x:round( (x/ df_vendor['ITM_TOT_AM'].sum())*100),2)
            df_vendor = df_vendor.iloc[:3,:]
            return df_vendor

        data1 = data[['COMMODITY','QUANTITY','ITM_TOT_AM','VENDOR_CODE','Category_Grouping']]
        df_vendor=data1.groupby(['Category_Grouping']).apply(transformer)

        x = list(set(data.Category_Grouping))
        color_list=['orange','green','red','blue','pink','maroon','black','#C25CEB','yellow','#EB755C']
        cnt=0
        data2=[]

        for i in x:

            langs = list(df_vendor.loc[i,:]['VENDOR_CODE'])
            students = list(df_vendor.loc[i,:]['percent'])
            values1= list(df_vendor.loc[i,:]['ITM_TOT_AM'])
            trace = go.Bar(
               x = langs,
               y = students,
                marker=dict(color=color_list[cnt]),
                name = i


            )
            cnt+=1
            data2.append(trace)


        layout = dict(
                          xaxis= dict(showgrid=False,title='Vendors'),
                          yaxis = dict(title= 'Total Amount as a Percentage',ticklen=5,zeroline=False,showgrid=False),
                          width=900,
                        height=700,
                       paper_bgcolor='rgba(0,0,0,0)',
              plot_bgcolor='rgba(0,0,0,0)')

        fig = go.Figure(data=data2,layout=layout)
        st.plotly_chart(fig)

        st.markdown("## Top Vendors per Category: Total Order Quantity")
        def transformer1(df):
            df_vendor=df.groupby(['Category_Grouping','VENDOR_CODE'])['QUANTITY'].sum().reset_index().sort_values(by='QUANTITY',ascending=False)
            df_vendor['percent'] = df_vendor['QUANTITY'].apply(lambda x:round( (x/ df_vendor['QUANTITY'].sum())*100),2)
            df_vendor = df_vendor.iloc[:2,:]
            return df_vendor

        data1 = data[['COMMODITY','QUANTITY','ITM_TOT_AM','VENDOR_CODE','Category_Grouping']]
        df_vendor1=data1.groupby(['Category_Grouping']).apply(transformer1)
        x = list(set(data.Category_Grouping))
        color_list=['orange','green','red','blue','pink','maroon','black','#C25CEB','yellow','#EB755C']
        cnt=0
        data2=[]

        for i in x:

            langs = list(df_vendor1.loc[i,:]['VENDOR_CODE'])
            students = list(df_vendor1.loc[i,:]['percent'])
            trace = go.Bar(
               x = langs,
               y = students,
                marker=dict(color=color_list[cnt]),
                name = i
            )
            cnt+=1
            data2.append(trace)


        layout = dict(
                          xaxis= dict(showgrid=False,title='Vendors'),
                          yaxis = dict(title= 'Total Quantity as a Percentage',ticklen=5,zeroline=False,showgrid=False),
                          width=900,
                        height=700,
                       paper_bgcolor='rgba(0,0,0,0)',
              plot_bgcolor='rgba(0,0,0,0)')

        fig = go.Figure(data=data2,layout=layout)
        st.plotly_chart(fig)


    if a == 'ML Modelling':
        @st.cache(allow_output_mutation=True)
        def load_data():
            data = pd.read_csv('D:\\HCL_office\\Use_Case3_SpendClassification\\Final_Labelled_CatGrouping_Cleaned.csv',index_col=0)
            data = data.drop('content',axis=1)
            return data
        data = load_data()
        st.text("")
        st.markdown('''## Setting the Baseline Accuracy: Proportional Chance Criteria''')
        st.image("latex.png",width=900)
        state_counts = Counter(data['Category_Grouping'])
        df_state = pd.DataFrame.from_dict(state_counts, orient='index')
        num=(df_state[0]/df_state[0].sum())**2
        # st.write("Population per class: {}\n".format(df_state))
        st.success("1.25 times Proportion Chance Criterion, our baseline accuracy: {}%".format(round(1.25*100*num.sum()),2))

        tf = TfidfVectorizer(min_df=0.001,ngram_range=(1,1))
        tf_vect = tf.fit_transform(data['COMMODITY_DESCRIPTION'])
        X_tfidf = tf_vect.toarray()
        y_tfidf = data.Category_Grouping


        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_tfidf, test_size=0.20, random_state=42,stratify=y_tfidf)
        lr = pickle.load(open('lr.sav', 'rb'))
        st.markdown("## Trying out the various Classification Models")
        cat_level1 =['Logistic Regression','Support Vector Machines','Naive Bayes','Neural Networks']
        b=st.selectbox('Select ML/DL Algorithm', cat_level1,index=0)
        if b=='Logistic Regression':
            st.markdown('## Machine Learning using **Logistic Regression**')
            html_tag=('''
            <div style='text-align: justify;'>

            Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.
            Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.


             </div>''')
            st.markdown(html_tag,unsafe_allow_html=True)
            scr = round(lr.score(X_test, y_test),2)

            st.success("The test set accuracy achieved is:\n{:.2f}"

                  .format(scr))

            y_prob = lr.predict_proba(X_test)

            macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                              average="macro")
            weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                                 average="weighted")
            macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                              average="macro")
            weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                                 average="weighted")

            st.success("One-vs-Rest ROC AUC scores:\n{:.2f}"

                  .format(macro_roc_auc_ovr))
            lst = lr.predict(X_test)
            st.markdown("## Confusion Matrix")
            cm = confusion_matrix(y_target=y_test,
                          y_predicted=lst,
                          binary=False)
            classlabel = ['Agriculture','Clothing','Construction','Electrical','Healthcare','Industrial Parts','Office Supplies','Others','Technology','Utilities']
            plt,ax= plot_confusion_matrix(conf_mat=cm,show_normed=True,figsize=(12,18),class_names=classlabel)
            plt.set_figwidth(12)
            plt.set_figheight(15)
            plt.align_ylabels()
            # plt.box(pad_inches =0)
            st.pyplot(plt,transparent=True,pad_inches=0)

            st.markdown("## Model Interpretation Using Eli5")
            html_object=eli5.show_weights(lr, vec=tf, top=20)
            raw_html = html_object._repr_html_()
            components.v1.html(raw_html,width=1500,height=900)
        # st.markdown(html_tag,unsafe_allow_html=True)
        elif b=='Support Vector Machines':
            st.markdown("#### WIP")
        elif b== 'Neural Networks':
            st.markdown("#### WIP")

    return 0

# commodity_classify()
