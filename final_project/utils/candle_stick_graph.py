import plotly.graph_objects as go


def create_candlestick_graph(Date, Open, High, Low, Close, title):
    fig = go.Figure(data=
                    [go.Candlestick(x=Date,
                                    open=Open,
                                    high=High,
                                    low=Low,
                                    close=Close)]
                    )

    fig.update_layout(
        title=title,
        yaxis_title="Price ($)"
    )

    fig.show()
