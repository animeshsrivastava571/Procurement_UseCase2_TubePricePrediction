import streamlit as st
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import numpy as np
from gluonts.evaluation import Evaluator
import matplotlib.pyplot as plt
from plotly.offline import iplot
from pylab import rcParams


def forecast():

    st.markdown("## About DeepAR")
    st.markdown(
        """
        The **Amazon SageMaker DeepAR** forecasting algorithm is a supervised learning algorithm for forecasting 
        scalar (one-dimensional) time series using recurrent neural networks (RNN). Classical forecasting 
        methods, such as autoregressive integrated moving average (ARIMA) or exponential smoothing (ETS), 
        fit a single model to each individual time series. They then use that model to extrapolate the time 
        series into the future. The basic algorithm uses **Recurrent Neural Networks.**
        
        In many applications, however, you have many similar time series across a set of cross-sectional units. For example, 
        you might have time series groupings for demand for different products, server loads, and requests for webpages. 
        For this type of application, you can benefit from training a single model jointly over all of the time series. 
        DeepAR takes this approach. When your dataset contains hundreds of related time series, DeepAR outperforms the 
        standard ARIMA and ETS methods. You can also use the trained model to generate forecasts for new time series 
        that are similar to the ones it has been trained on.
        
        
        """
    )
    st.markdown('## Monthly Forecast using DeepAR')
    df = pd.read_csv('Warehouse_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    # df.info()
    # df.tail()

    df_ex = df.set_index('Date')
    custom_dataset = df_ex.to_numpy().T

    prediction_length = 12
    freq = "M"
    #custom_dataset = np.random.normal(size=(N, T))
    start = pd.Timestamp("31-01-2012", freq=freq)
    from gluonts.dataset.common import ListDataset
    train_ds = ListDataset([{'target': x, 'start': start}
                            for x in custom_dataset[:, :-prediction_length]],
                           freq=freq)

    test_ds = ListDataset([{'target': x, 'start': start}
                           for x in custom_dataset],
                          freq=freq)
    # @st.cache(suppress_st_warning=True)
    def train_depar(freq,prediction_length,train_ds):
        estimator = DeepAREstimator(freq= freq, prediction_length=prediction_length, trainer=Trainer(epochs=2))
        predictor = estimator.train(training_data=train_ds)
        return predictor

    predictor= train_depar(freq,prediction_length,train_ds)

    from gluonts.evaluation.backtest import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )


    forecasts = list(forecast_it)
    tss = list(ts_it)


    ts_entry = tss[0]
    np.array(ts_entry[:5]).reshape(-1,)

    def plot_prob_forecasts(ts_entry, forecast_entry, wareh_name):
        plot_length = 150
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

        fig, ax = plt.subplots(1, 1, figsize=(30 ,16))
        ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='r')
        plt.grid(which="major")
        plt.legend(legend, loc="upper left",fontsize=40)
        plt.title(wareh_name,fontsize=50)
        plt.xlabel('Date',fontsize=40)
        plt.ylabel('Order Demand',fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        st.pyplot()


    ts_entry = tss[0]
    forecast_entry = forecasts[0]


    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse A Forecast' )
    ts_entry = tss[1]
    forecast_entry = forecasts[1]
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse C Forecast' )
    # print("*************************************\n")
    # print(forecast_entry.)


    ts_entry = tss[2]
    forecast_entry = forecasts[2]
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse J Forecast' )


    ts_entry = tss[3]
    forecast_entry = forecasts[3]
    plot_prob_forecasts(ts_entry, forecast_entry, 'Warehouse S Forecast' )


    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))


    metric_df = item_metrics[['MAPE']]
    list_w = ['Warehouse A', 'Warehouse C', 'Warehouse J', 'Warehouse S']
    se = pd.Series(list_w)
    metric_df['Warehouse'] = se.values

    st.markdown("## Look at the MAPE values for different warehouses")
    st.table(metric_df)
    print("Success")

# forecast()



