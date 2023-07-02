import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Function to make the flight amount prediction
def Prediction(inp):
    X = []
    for idx, i in enumerate(inp):
        if columns[idx] in log_cols:
            X.append(np.log(i)+1e-2)
        elif columns[idx] == "Status":
            X.append(enc.transform([i]))
        else:
            X.append(i)

    X = Scaler.transform([X])
    return Model_random_forest.predict(X)[0]

def country_names():
    file = open('Country.txt')
    lst = []
    lr = ' '
    while lr:
        lr = file.readline().strip()
        if lr:
          lst.append(lr)
    return lst

def plot_graph(colmn):
    buttons = []
    i = 0
    fig3 = go.Figure()

    for country in country_list:
        fig3.add_trace(
            go.Scatter(
                x=graph['Year'],
                y=graph[graph["Country"] == country][colmn],
                name=country,
                visible=(i == 0),
            )
        )

    for country in country_list:
        args = [False] * len(country_list)
        args[i] = True

        # Create a button object for the country we are on
        button = dict(
            label=country,
            method="update",
            args=[{"visible": args}],
        )

        # Add the button to our list of buttons
        buttons.append(button)

        # i is an iterable used to tell our "args" list which value to set to True
        i += 1

    fig3.update_layout(
        updatemenus=[
            dict(
                active=0,
                type="dropdown",
                buttons=buttons,
                x=0.5,
                y=1.1,
                xanchor="center",
                yanchor="top",
                bgcolor="white",
                bordercolor="black",
                font=dict(color="black"),
            ),
        ],
        title=dict(
            text=colmn + " Comparison by Country",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
        xaxis=dict(
            title="Year",
            showline=True,
            linewidth=1,
            linecolor="white",
            mirror=True,
        ),
        yaxis=dict(
            title=colmn,
            showline=True,
            linewidth=1,
            linecolor="white",
            mirror=True,
        ),
        autosize=False,
        width=800,
        height=600,
        plot_bgcolor="black",
        paper_bgcolor="black",
    )

    # fig3.show()
    return fig3

def country_plot(colmn):
    fig = px.choropleth(
        graph,
        locations="iso",
        color=colmn,
        hover_name="Country",
        color_continuous_scale=px.colors.sequential.Plasma,
        animation_frame="Year",
    )

    fig.update_layout(
        title=dict(
            text="Choropleth Map: " + colmn,
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top"
        ),
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth"
        ),
        coloraxis_colorbar=dict(
            title=colmn,
            thickness=20,
            len=0.5,
            x=0.92,
            y=0.45,
            yanchor="middle",
            ticks="outside",
            tickfont=dict(size=10)
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black"
    )

    # fig.show()
    return fig 


# Load the trained model
pickle_in = open("model.pkl", "rb")  ## rb = READ BYTE
Model_random_forest = pickle.load(pickle_in)

pickle_in = open("encoder.pkl", "rb")  ## rb = READ BYTE
enc = pickle.load(pickle_in)

pickle_in = open("scaler.pkl", "rb")  ## rb = READ BYTE
Scaler = pickle.load(pickle_in)

file = open('columns.txt')
columns = [x.strip() for x in file.readline().replace("'", "")[1:-1].split(',')]
log_cols = [x.strip() for x in file.readline().replace("'", "")[1:-1].split(',')]
num_col = [x.strip() for x in file.readline().replace("'", "")[1:-1].split(',')][1:]
country_list = country_names()
graph = pd.read_csv('Plots.csv')
# print(columns, log_cols, num_col)

# Streamlit app
def main():
    st.title("Life Expectancy Prediction")
    st.subheader("History")

    which = st.selectbox("Column", num_col)

    # Create two columns
    # col1, col2 = st.columns([1,2], gap="large")

    # Plot graph 1 in the first column
    # with col1:
    fig1 = plot_graph(which)
    st.plotly_chart(fig1)

    # Plot graph 2 in the second column
    # with col2:
    fig2 = country_plot(which)
    st.plotly_chart(fig2)

    # Input features
    st.subheader("Fill the current details based on your Nationality ")
    status = st.selectbox("Status", ["Developing", "Developed"])
    am = st.number_input("Adult Mortality", min_value=1, max_value=800)
    ifd = st.number_input("Infant deaths", min_value=0.1, max_value=1800.00, step=10.0)
    alc = st.number_input("Alcohol", min_value=0.01, max_value=30.0, step=0.1)
    ms = st.number_input("Measles", min_value=0.1, max_value=300000.0, step = 1000.0)
    bmi = st.number_input("Country's average BMI", min_value=1, max_value=90)
    pol = st.number_input("Country's Polio Status", min_value=3, max_value=100)
    ad = st.number_input("HIV/AIDS", min_value=0.1, max_value=60.0)
    thn = st.number_input("Thinness 1-19 years", min_value=0.1, max_value=30.0, step=0.1)
    icr = st.number_input("Income composition of resources", min_value=0.1, max_value=1.0, step=0.1)

    # Preprocess input
    inp = [status, am, ifd, alc, ms, bmi, pol, ad, thn, icr]
    
    # Make prediction
    prediction = Prediction(inp)

    # Display the prediction
    st.subheader("Life Expectancy Prediction")
    st.write(f"Expected Life of a person in your country is: {prediction} years !!")


if __name__ == '__main__':
    main()
