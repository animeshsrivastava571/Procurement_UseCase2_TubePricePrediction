import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# Plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode()
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, GridSearchCV
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.metrics import r2_score
import os
from PIL import Image

def predict_price():

    img = Image.open('tb_img1.png')
    st.image(img,width=900)
    st.markdown("## Feature Engineering")
    img = Image.open('tb_img2.png')
    st.image(img,width=900)
    file_csv=[]
    for f in os.listdir("."):
        if f.endswith('.csv'):
            file_csv.append(f)
    selected_filename = st.selectbox('Load a test file to execute',file_csv,index=4)
    st.write('You selected `%s`' % selected_filename + '. To perform analysis on this file, continue below')
    df_test = pd.read_csv(selected_filename,nrows=10000)

    num_cols = ['annual_usage', 'diameter', 'wall', 'length', 'num_bends',
       'bend_radius', 'num_boss', 'num_bracket', 'other', 'spec_totals',
       'type_totals', 'component_totals', 'revised_quantity','newCol','weight', 'thickness', 'comp_length',
          'id','num_days_supp_relationship']



    df_test_num = df_test[num_cols]
    corr_df =df_test_num.corr()
    st.markdown("## Correlation Heat Map")
    # Plot the correlation heatmap
    data = [go.Heatmap(z=corr_df.values,
                                  colorscale='Blackbody',
                                   x=list(corr_df.columns),
                                   y=list(corr_df.index))
                                  ]


    layout = go.Layout(
        autosize=False,

        width=900,
        height=700,

            xaxis=dict(
            title='',
            showgrid=False,
            titlefont=dict(
               # family='Gill sans, monospace',
                size=12,
                #color='#7f7f7f'
            ),
            showticklabels=True,
            tickangle=90,
            tickfont=dict(
                family="Gill Sans MT",
                size=12,
                color='black'
            ),
        ),
        yaxis=dict(
            title='',
            showgrid=False,

            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                family="Gill Sans MT",
                size=12,
                color='black'
            ),
    )
    )

    fig = dict(data = data, layout = layout)
    # iplot(fig)
    st.plotly_chart(fig)

    st.markdown('## Relationship between Days of Conversion and Price')
    df3 = df_test[['num_days_supp_relationship','newCol']]

    def numday_tomonyear(days):
        if (days < 90):
            return "within_three"
        elif(90 > days < 365):
            return "within year"
        elif(365 < days < 1095):
            return "within three years"
        else:
            return "more than three years"
    df3['convertion'] = df3.apply(lambda x: numday_tomonyear(x['num_days_supp_relationship']), axis = 1)
    df3=df3.rename({'newCol':'price'},axis=1)
    ax= sns.catplot('convertion', 'price', data=df3, aspect = 1.5)
    st.pyplot(ax)

    # Load a RF Model
    loaded_model = pickle.load(open('rf_model.sav', 'rb'))
    df_test1 = df_test.loc[:, df_test.columns != 'newCol']
    df_test1.head()
    arr1=np.array(df_test1)
    df_test['predicted_value']=loaded_model.predict(arr1)
    st.markdown('## Machine Learning using Random Forest Regression')
    st.markdown('''**Random forest** is a Supervised Learning algorithm which uses ensemble learning 
    method for classification and regression.Random forest is a bagging technique and not a boosting technique. 
    The trees in random forests are run in parallel. There is no interaction between these trees while building the 
    trees.It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
    A random forest is a meta-estimator (i.e. it combines the result of multiple predictions) which aggregates many 
    decision trees, with some helpful modifications:''')

    # st.markdown('Press **Run** to run the ML model on the uploaded data')
    st.write('You have selected `%s`' % selected_filename + '. Press **Run** to execute the ML model on the uploaded data')
    # button1 = st.button('Run')
    # st.markdown(":chart_with_upwards_trend:")
    if st.button('Run'):
        st.text('R2 score:')
        st.success(round(r2_score(df_test['predicted_value'], df_test['newCol']),3))
        st.markdown("### Looking at Some of the Actual and the Predicted values")
        st.text("")
        df_12 = df_test[['newCol','predicted_value']].iloc[:5,:]
        df_12 = df_12.rename({'newCol':'actual_value'},axis=1)
        st.table(df_12)

        trace0 = go.Scatter(
        x =df_test['predicted_value'],
        y = df_test['newCol'],
        mode = 'markers',
        name = 'Predicted vs Actual Values',
        text= df_test['predicted_value']
         )

        data = [trace0]


        layout = dict(title = 'Predicted vs Actual Values',title_x=0.5,
                          xaxis= dict(title= 'Predicted Values',ticklen= 5,zeroline= False),
                          yaxis = dict(title= 'Actual Values',ticklen=5,zeroline=False),
                          width=900,
                        height=700)

        fig = dict(data = data, layout = layout)
        st.plotly_chart(fig)

        # st.markdown("Th")
        X = pd.read_csv('columns.csv',index_col=0)

        ranks = sorted(list(zip(loaded_model.feature_importances_, X.columns)), key=lambda x: -x[0])
        ranks=ranks[:10]
        df_ranks = pd.DataFrame(ranks,columns=['values1','quantity'])

        langs = list(df_ranks.quantity)
        students = list(df_ranks.values1)
        data = [go.Bar(
           x = langs,
           y = students
        )]
        layout = dict(title = 'Feature Importances',title_x=0.5,
                          xaxis= dict(title= 'Features',ticklen= 5,zeroline= False),
                          yaxis = dict(title= 'Importances',ticklen=5,zeroline=False),
                          width=900,
                        height=700)
        fig = go.Figure(data=data,layout=layout)
        st.plotly_chart(fig)


    return 0

# predict_price()
