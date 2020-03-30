import datetime
import os
import yaml

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize



import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output



# Lecture du fichier d'environnement
ENV_FILE = '/Users/EnzoButhiot/Documents/corona/env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

# Lecture du fichier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df: _df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020, 3, 21)]
              )

korea_df =(epidemie_df[epidemie_df['Country/Region']=='South Korea']
        .groupby(['Country/Region','day'])
        .agg({'Confirmed':'sum', 'Deaths':'sum','Recovered':'sum'})
        .reset_index()
          )

korea_df['infected']= korea_df['Confirmed'].diff()


countries = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]


app = dash.Dash('Corona Virus Explorer')
app.layout = html.Div([
    html.H1(['Corona Virus Explorer'], style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
            html.Div([
                dcc.Dropdown(
                    id='country',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country2',
                    options=countries
                )
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable',
                    options=[
                        {'label': 'Confirmed', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')
            ]),   
        ]),
        dcc.Tab(label='Map', children=[
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                #marks={i:str(date) for i, date in enumerate(epidemie_df['day'].unique())}
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )  
        ]),
        dcc.Tab(label='SIR model', children=[
            html.Div([
                dcc.Dropdown(
                    id='country3',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Input(id='Beta', value= 0.001, type='number', debounce = True),
                html.Div(id='my-div')
            ]),
            html.Div([
                dcc.Input(id='Gamma', value= 0.1, type='number', debounce= True),
                html.Div(id='my-div2')
            ]),
            html.Div([
                dcc.Input(id='Population', value= 51_470_000 , type='number', debounce=True),
                html.Div(id='my-div3')
            ]),
            html.Div([
                dcc.RadioItems(
                    id='parameters',
                    options=[
                        {'label':'No optimized parameters', 'value':'No optimized parameters'},
                        {'label':'Optimized parameters', 'value':'Optimized parameters'}],
                    value= 'No optimized parameters',
                    labelStyle={'display': 'inline-block'}
                ) 
            ]),
            html.Div([
                 html.Div(id="my-div4")    
            ]),
            html.Div([
                dcc.Graph(id='graph2')
            ])
        ]),
    ]),
])

@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),
        Input('country2', 'value'),
        Input('variable', 'value'),        
    ]
)
def update_graph(country, country2, variable):
    print(country)
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )
    if country2 is not None:
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                     .groupby(['Country/Region', 'day'])
                     .agg({variable: 'sum'})
                     .reset_index()
                    )

        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )            
        ] if country2 is not None else [])
    }

@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day', 'value'),
    ]
)
def update_map(map_day):
    day = epidemie_df['day'].unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({'Confirmed': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )
    print(map_day)
    print(day)
    print(map_df.head())
    return {
        'data': [
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + ' (' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed'] / 1_000, 5)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True),
        )
    }

@app.callback(
    [Output(component_id='my-div', component_property='children'),
     Output(component_id='my-div2', component_property='children'),
     Output(component_id='my-div3', component_property='children')
    ],
    [Input(component_id='Beta', component_property='value'),
     Input(component_id='Gamma', component_property='value'),
     Input(component_id='Population', component_property='value')
    ]
)

def update_output_div(Beta, Gamma, Population):
    Beta = Beta
    Gamma = Gamma
    Population = Population
    return "Beta={}".format(Beta), 'Gamma={}'.format(Gamma), 'Population={}'.format(Population)


@app.callback(

    Output("graph2", "figure"),
[  Input ("Beta", "value"),
   Input("Gamma", "value"),
   Input("Population", "value"),
   Input("country3", "value")
])

def sol(Beta, Gamma, Population, country):
    
    def SIR(t,y):
        S=y[0]
        I=y[1]
        R=y[2]
        return([-Beta*S*I, Beta*S*I-Gamma*I, Gamma*I])
    
    print(country)
    
    epidemie_df2 = (epidemie_df[epidemie_df['Country/Region']==country]
        .groupby(['Country/Region','day'])
        .agg({'Confirmed':'sum', 'Deaths':'sum','Recovered':'sum'})
        .reset_index()
          )
    epidemie_df2['infected'] = epidemie_df2['Confirmed'].diff()
    
    solution = solve_ivp(SIR, [0, 40], [Population*0.00001, 1, 0], t_eval=np.arange(0, 40, 1))
    
    #On multiplie la population par 0.0001 car sinon le solveur met trop de temps et les courbes s'affichent très tard
    
    df_solution = (pd.DataFrame(data=[solution.t, solution.y[0], solution.y[1], solution.y[2] ])
                      .T
                      .rename(columns={0: "Time", 1: "Susceptible", 2 : "Infected", 3: "Recovered"}))
    
    
    return{
        'data' : [
            dict(
            x = df_solution['Time'],
            y = df_solution["Susceptible"],
            type="line",
            name = "Susceptible"
            )
        ] + ([
            dict(
            x = df_solution['Time'],
            y = df_solution['Infected'],
            type="line",
            name = "Infected"
            )
        ]) + ([
            dict(
            x = df_solution['Time'],
            y = df_solution['Recovered'],
            type="line",
            name = "Recovered"
            )
        ]) + ([
            dict(
            x = epidemie_df2.loc[2:]['infected'].reset_index(drop=True).index,
            y = epidemie_df2.loc[2:]['infected'],
            type="line",
            name = "Original data"
            )
        ])
    }



    


if __name__ == '__main__':
    app.run_server(debug=True)