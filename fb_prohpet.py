import pandas as pd
import streamlit as st
from fbprophet import Prophet
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()

fig = make_subplots(rows=1, cols=2)

st.title('Summary')
df_final = pd.read_csv('Ani1.csv',parse_dates=['Date'],infer_datetime_format=True)
df_final= df_final.drop('Unnamed: 0',axis=1)
df_warehouse_j = df_final[['Date', 'Order_Demand_Whse_J']]
df_j = df_warehouse_j.rename(columns={"Date": "ds", "Order_Demand_Whse_J": "y"})
prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,)
prophet.fit(df_j)
future = prophet.make_future_dataframe(periods=12, freq='M')
df_forecast = prophet.predict(future)


trace = go.Scatter(
    name = 'Actual Demand Quantity',
    mode = 'markers',
    x = list(df_j['ds']),
    y = list(df_j['y']),

    marker=dict(
        color='#FFBAD2',
        line=dict(width=1),
        size=2
    )
)
fig.add_trace(trace,row=1,col=1)

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),

    marker=dict(
        color='red',
        line=dict(width=10)
    )
)

fig.add_trace(trace1,row=1,col=1)

upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),

    line= dict(color='#57b88f'),
    fill = 'tonexty'
    # fillcolor='#E0F2F3',
    # opacity=0.1
)
fig.add_trace(upper_band,row=1,col=1)

lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',

    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)

fig.add_trace(lower_band,row=1,col=1)

tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df_j['ds']),
   y = list(df_j['y']),

   marker=dict(
      color='black',
      line=dict(width=2)
   )
)

fig.add_trace(tracex,row=1,col=1)

data = [trace1, lower_band, upper_band, trace]

layout = dict(title='Order Demand Forecasting - Warehouse J',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
              yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
              width=900,
            height=500,
             )

fig=dict(data=data,layout=layout)
# plt.savefig('btc03.html')
st.plotly_chart(fig)




fig = make_subplots(rows=1, cols=2)
trace = go.Scatter(
    name = 'Actual Demand Quantity',
    mode = 'markers',
    x = list(df_j['ds']),
    y = list(df_j['y']),

    marker=dict(
        color='#FFBAD2',
        line=dict(width=1),
        size=2
    )
)
fig.add_trace(trace,row=1,col=2)

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),

    marker=dict(
        color='red',
        line=dict(width=10)
    )
)

fig.add_trace(trace1,row=1,col=2)

upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),

    line= dict(color='#57b88f'),
    fill = 'tonexty'
    # fillcolor='#E0F2F3',
    # opacity=0.1
)
fig.add_trace(upper_band,row=1,col=2)

lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',

    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)

fig.add_trace(lower_band,row=1,col=2)

tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df_j['ds']),
   y = list(df_j['y']),

   marker=dict(
      color='black',
      line=dict(width=2)
   )
)

fig.add_trace(tracex,row=1,col=2)

data = [trace1, lower_band, upper_band, trace]

layout = dict(title='Order Demand Forecasting - Warehouse J',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
              yaxis=dict(title = 'Order Quantity', ticklen=2, zeroline=True),
              width=900,
            height=500,
             )

fig=dict(data=data,layout=layout)
# plt.savefig('btc03.html')
st.plotly_chart(fig)
