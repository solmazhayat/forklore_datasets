import plotly.graph_objects as go
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State,no_update
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import requests

# "C:\Users\akkus\OneDrive\Masaüstü\PROJ201\Final\Code\Forklore_final.py"

df = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/Finalized_KamerHoca_dataset.csv")
ob1 = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/Obese%20Data.csv")
ob2 = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/obsdfinal.csv", sep=';')
co2_df = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/historical_emissions2.csv")
country_to_region_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/final.csv",  delimiter=';')
protein_df = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/protein.csv')
carbohydrate_df = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/carbohydrate.csv')
fat_df = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/saturated_fat.csv')
calcium_dfo = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/calcium.csv')
number_of_ing_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/top_5_ingredients_by_region.csv")
region_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/country_region.csv")
affordibility_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/faostatnewv.csv" ,  delimiter=';')

logo="https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/proj.jpg"

obeseData = ob1
obeseDataGN = obeseData[obeseData['Gender'] == 'Total']
obeseDataGNonlyCount = obeseDataGN.dropna(subset=['Country Name'])

obeseDataGNonylCountAdult = obeseDataGNonlyCount[obeseDataGNonlyCount['Indicator Name'] == 'Obesity, adults aged 18+']


obeseDataWRegion = ob2

obeseDataWRegionGN = obeseDataWRegion[obeseDataWRegion['Gender'] == 'Total']


obeseDataWRegionAdult = obeseDataWRegion[obeseDataWRegion['Indicator Name'] == 'Obesity, adults aged 18+']

obeseDataWRegionAdultGendered = obeseDataWRegionAdult[obeseDataWRegionAdult['Gender'] != 'Total']

#start of the code by İdil/Hayat
consumption_column = 'Median'

# Define a dictionary to map nutrient filenames (modify as needed)
nutrient_files = {
  'Protein': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v23_totprotein.csv',
  'Carbohydrate': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v22_totcarbs.csv',
  'Fiber': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v34_fiber.csv',
  'Fruit': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v01_fruits.csv',
  'Beans and legumes': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v05_beans.csv',
  'Nuts and Seeds': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v06_nuts.csv',
  'Saturated Fats': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v27_satfat.csv',
  'Sodium': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v37_sodium.csv',
  'Iron': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v39_iron.csv',
  'Vitamin C': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v51_vitC.csv',
  'Vitamin D': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v52_vitD.csv',
  'Vitamin E': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v53_vitE.csv',
  'Vitamin B12': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v50_cnty.csv',
  'Vitamin A': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v43_vitA.csv',
  'Calcium': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/v36_cnty.csv',
}

# Define a dictionary to map nutrient filenames for radar chart data (modify as needed)
radar_nutrient_files = {
  'Protein': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/protein.csv',
  'Carbohydrate': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/carbohydrate.csv',
  'Fiber': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/fiber.csv',
  'Fruit': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/fruit.csv',
  'Beans and legumes': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/Beans_and_legumes.csv',
  'Nuts and Seeds': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/Nuts.csv',
  'Saturated Fats': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/saturated_fat.csv',
  'Sodium': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/sodium.csv',
  'Iron': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/iron.csv',
  'Vitamin C': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/vitaminC.csv',
  'Vitamin D': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/vitaminD.csv',
  'Vitamin E': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/vitaminE.csv',
  'Vitamin B12': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/vitaminB12.csv',
  'Vitamin A': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/vitaminA.csv',
  'Calcium': 'https://raw.githubusercontent.com/solmazhayat/forklore_datasets/new/calcium.csv',

}

# Function to load data from a CSV file
def load_fruit_data(filename):
  df = pd.read_csv(filename)
  return df

# Function to create the pie chart figure
def create_pie_chart(df_filtered, nutrient, selected_countries):
  # Define a dictionary to map display labels to nutrient names
  nutrient_labels = {
    'Protein': 'Protein',
    'Carbohydrate': 'Carbohydrate',
    'Fiber': 'Fiber',
    'Fruit': 'Fruit',
    'Beans and legumes': 'Beans and Legumes',
    'Nuts and Seeds': 'Nuts and Seeds',
    'Saturated Fats': 'Saturated Fats',
    'Sodium': 'Sodium',
    'Iron': 'Iron',
    'Vitamin C': 'Vitamin C',
    'Vitamin D': 'Vitamin D',
    'Vitamin E': 'Vitamin E',
    'Vitamin B12': 'Vitamin B12',
    'Vitamin A': 'Vitamin A',
    'Calcium': 'Calcium',
    
  }
  
  df_filtered = df_filtered[df_filtered['country name'].isin(selected_countries)]
  fig = go.Figure(data=[go.Pie(labels=df_filtered['country name'],
                                values=df_filtered[consumption_column],
                                hole=0.3)])
  fig.update_layout(title=f"Average {nutrient_labels[nutrient]} Consumption (Max 6 Countries)")
  return fig

# Function to create the radar chart figure
def create_radar_chart(df_filtered, nutrient, selected_regions):
  # Filter data for the selected region
  df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]

  # Get unique regions from the filtered DataFrame
  unique_regions = df_filtered['Region'].unique()

  # Create radar chart data
  fig = go.Figure()
  for region in unique_regions:
    # Filter data for the specific region
    region_data = df_filtered[df_filtered['Region'] == region]
    categories = region_data['Year'].astype(str).tolist()  # Years as categories
    values = region_data[consumption_column].tolist()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=region
        )
    )

  fig.update_layout(
      title=f"Average {nutrient} Consumption in Selected Regions Over Time",
      polar=dict(
          radialaxis=dict(visible=True, range=[0, max(df_filtered[consumption_column]) * 1.1]),  # Adjust range for better visibility
      ),
      showlegend=True
  )
  return fig

# Load initial data (replace with your filenames and structure)
dataframes = {}
for nutrient, filename in nutrient_files.items():
  df = load_fruit_data(filename)
  # Assuming all files have the same structure with "country name" column
  dataframes[nutrient] = df

# Load radar data for each nutrient
radar_dataframes = {}
for nutrient, filename in radar_nutrient_files.items():
  df = pd.read_csv(filename)
  df['Year'] = df['Year'].astype(int)  # Ensure 'Year' column is numeric
  radar_dataframes[nutrient] = df

# Define available nutrients (modify as needed)
nutrient_options = ['Protein','Carbohydrate', 'Fiber','Fruit','Beans and legumes','Nuts and Seeds','Saturated Fats','Sodium','Iron','Vitamin C','Vitamin D','Vitamin E','Vitamin B12','Vitamin A','Calcium']

#end of the code by İdil/Hayat

calcium_df = calcium_dfo.copy()


calcium_df["Median"] = calcium_df["Median"] / 10


protein_df['Nutrient'] = 'Protein (g)'
carbohydrate_df['Nutrient'] = 'Carbohydrate (g)'
fat_df['Nutrient'] = 'Fat (g)'
calcium_df['Nutrient'] = 'Calcium (0,1mg)'


combined_df = pd.concat([protein_df, carbohydrate_df, fat_df, calcium_df])

fig = go.Figure(data=go.Choropleth(
    locations=co2_df['Country'][1:],
    locationmode='country names',
    z=co2_df['2020'][1:],
    text="",
    colorscale='fall',
    autocolorscale=False,
    reversescale=False, 
    marker_line_color='darkgray',
    marker_line_width=1,
    colorbar_tickprefix='',
    colorbar_title='CO2 release (kg/L)',
))

fig.update_layout(
    width=1500,  
    height=500,  
    margin={"r":0,"t":0,"l":0,"b":0}  
)



first = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = first.server

#----------------------------------------------------------------------------------------------------------------------------------------


iceFig = px.icicle(
    obeseDataGNonylCountAdult,
    path=[px.Constant("World"), 'Region', 'Country Name'],
    values='Numeric',
    color='Numeric',
    color_continuous_scale='RdBu_r',
    title='% of obesity among adults',
    labels={'Numeric': ''},
)

iceFig.update_layout(font=dict(size=16))
iceFig.update_traces(hovertemplate='', hoverinfo='skip')


def create_age_histogram(data):
    fig = px.histogram(data, x='Indicator Name', y='Numeric', histfunc='avg', labels={'Numeric': 'percentage', 'Indicator Name': ''})

    fig.update_layout(font=dict(size=16))


    values = data['Numeric'].values
    sorted_values = sorted(values)
    color_map = {sorted_values[0]: '#2971b1', sorted_values[len(sorted_values) // 2]: '#fac8af', sorted_values[-1]: '#b6212f'}
    colors = [color_map[val] for val in values]

    fig.update_traces(hovertemplate='', hoverinfo='skip', marker=dict(color=colors))
    return fig


ageFig = create_age_histogram(obeseDataWRegionGN[obeseDataWRegionGN['Country Name'] == 'World'])


def create_gender_histogram(data):
    fig = px.histogram(data, x='Gender', y='Numeric', histfunc='avg', labels={'Numeric': 'percentage', 'Gender': ''})

    fig.update_layout(font=dict(size=16))


    values = data['Numeric'].values
    sorted_values = sorted(values)
    color_map = {sorted_values[0]: '#2971b1', sorted_values[-1]: '#b6212f'}  
    colors = [color_map[val] for val in values]

    fig.update_traces(hovertemplate='', hoverinfo='skip', marker=dict(color=colors))
    return fig


genderFig = create_gender_histogram(obeseDataWRegionAdultGendered[obeseDataWRegionAdultGendered['Country Name'] == 'World'])

#----------------------------------------------------------------------------------------------------------------------------------------

items = [
    'Obesity', 'Overweight'
]

CSE_ID = "87b06a24b639a4481"
API_KEY = "AIzaSyCPujz53FZgvdOpSZ8Xk81rKt1qYO8wdp0"

#----------------------------------------------------------------------------------------------------------------------------------------

first.layout = dbc.Container([
    dbc.Col([
        dbc.Row([
            dbc.Col([
                    html.H1("Forklore", style={'text-align': 'left', 'color': 'black'}),
                    html.H3("A nutritional data visualization web-app.", style={'text-align': 'left', 'color': 'black'})
        ]),
            dbc.Col(html.Img(src=logo, style={'width': '100px', 'height': '100px'}), width=2)
                 ]),      
        html.Hr(),
        dbc.Row([
            html.Div(
                children=[
                    html.H2(children="Get Links for your Recipes!"),
                    dcc.Input(
                    id="search-input",
                    placeholder="Enter your recipe query...",
                    type="text",
                    value="",
                    style={'width': '75%', 'padding': '10px'}
                    ),
                    html.Button(
                    id="search-button",
                    children="Search",
                    style={'padding': '10px'}
                    ),
                    html.Div(
                    id="linkresults",
                    children=[]
                    )
                ]
            )
        ], style={'padding-bottom': '50px'}),
        dbc.Row([
            html.H2(children='Match Foods with their regions!'),
            dcc.Input(id='search-bar', type='text', placeholder='Search for a recipe or region...', style={'width': '75%', 'padding': '10px'}),
            html.Div(id='results')
        ], style={'padding-bottom': '50px'}),
        dbc.Row([
            html.H2(children='Obesity Data from around the World'),

        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=iceFig, id="iceGraph", style={"height": "1000px"}), width=6),
            dbc.Col([
                dcc.Graph(figure=ageFig, id="ageGraph", style={"padding-top": "50px"}),
                dcc.Graph(figure=genderFig, id="genderGraph", style={'padding-top': '50px'})
            ], width=6),
        ]),
            html.Hr(),
        dbc.Row([
            html.H2(children='CO2 Emissions of the Countries Around the World in 2020'),
        ]),
        dbc.Row([
            dbc.Col(html.Div(
                [
                dcc.Graph(figure=fig,id="co2map")
                ]
            ))
        ], style={}),
        
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(id="fruit"),
                            dcc.Slider(
            protein_df['Year'].min(),
            protein_df['Year'].max(),
            step=None,
            id='year-slider',
            value=protein_df['Year'].max(),
            marks={str(year): str(year) for year in protein_df['Year'].unique()})])),
            dbc.Col(dcc.Graph(id="co2")),
            dbc.Col(dcc.Graph(id="ingridient"))
        ]),
        dbc.Row([
            html.Div(id="economy" , style={'text-align': 'right'})
        ]),
        html.Hr(),
        dbc.Row([
            html.H2(children='Nutritional Data Comparison Among Countries(Max 6)'),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='nutrient-dropdown',
                    options=[{'label': nut, 'value': nut} for nut in nutrient_options],
                    value='Fruit',  # Set initial selection
                    placeholder="Select a nutrient..."
                ),
                dcc.Dropdown(
                    id='country-dropdown',
                    multi=True,  # Allow selecting multiple countries
                    options=[],  # Options will be populated later
                    value=['Spain', 'China', 'Turkey'],  # Set initial selection
                    placeholder="Select up to 6 countries..."
                ),
                dcc.Graph(
                    id='pie-chart',
                    className='six columns'
                ),
                dcc.Graph(
                    id='radar-chart',
                    className='six columns'
                ),
            ])
        ]),  
    ])
])

#----------------------------------------------------------------------------------------------------------------------------------------

@callback(
    Output(component_id='ageGraph', component_property='figure'),
    Input(component_id='iceGraph', component_property='clickData'),
)
def updateAgeGraph(regionChosen):
    if regionChosen is not None:
        region = regionChosen['points'][0]['label']
        filtered_data = obeseDataWRegionGN[obeseDataWRegionGN['Country Name'] == region]
    else:
        filtered_data = obeseDataWRegionGN[obeseDataWRegionGN['Country Name'] == 'World']

    return create_age_histogram(filtered_data)


@callback(
    Output(component_id='genderGraph', component_property='figure'),
    Input(component_id='iceGraph', component_property='clickData'),
)
def updateGenderGraph(regionChosen):
    if regionChosen is not None:
        region = regionChosen['points'][0]['label']
        filtered_data = obeseDataWRegionAdultGendered[obeseDataWRegionAdultGendered['Country Name'] == region]
    else:
        filtered_data = obeseDataWRegionAdultGendered[obeseDataWRegionAdultGendered['Country Name'] == 'World']

    return create_gender_histogram(filtered_data)


@callback(
    Output('results', 'children'),
    [Input('search-bar', 'value')]
)
def update_results(search_value):
    if not search_value:
        return []
    
    search_value = search_value.lower()


    if 'Recipe Name' not in df.columns or 'Region' not in df.columns:
        return [html.Div("Columns 'Recipe Name' or 'Region' not found in the dataset.")]

    filtered_df = df[df.apply(lambda row: search_value in str(row['Recipe Name']).lower() or search_value in str(row['Region']).lower(), axis=1)]
    if filtered_df.empty:
        return [html.Div("No results found.")]

    results_to_display = filtered_df.head(10)

    results = [html.Div(f"{row['Recipe Name']} - {row['Region']}") for _, row in results_to_display.iterrows()]

    if len(filtered_df) > 10:
        results.append(html.Div('More results available...'))

    return results

@callback(
    Output("linkresults", "children"),
    [Input("search-button", "n_clicks")],
    [State("search-input", "value")]
)
def update_results(n_clicks, search_query):
    if n_clicks is None:
        return []
    else:
        search_query = search_query + " recipe" 
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CSE_ID}&q={search_query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:  
                first_result = data["items"][0]
                return [html.A(href=first_result["link"], children=first_result["title"])]
            else:
                return [html.P("No results found")]
        else:
            return [html.P("Error: Could not retrieve search results")]

@callback(
    Output(component_id='fruit', component_property='figure'),
    Output(component_id='co2', component_property='figure'),
    Output(component_id='ingridient', component_property='figure'),
    Output(component_id='economy', component_property='children'),
    Input(component_id='year-slider', component_property='value'),
    Input(component_id='co2map', component_property='clickData'),
    prevent_initial_call=True
)
def update_children(year_selected,click):
    if click is None:
        return no_update,no_update,no_update,no_update
    else:
        country=click["points"][0]["location"]
        region_row=country_to_region_df[country_to_region_df["Country"]==country]
        region = region_row["Subregion"].iloc[0]
        filtered_data = combined_df[(combined_df['Year'] == year_selected) & (combined_df['Region'] == region)]
        row = co2_df[co2_df['Country'] == country]
        
        continent_row=region_df[region_df["Unnamed: 2"]==country]


        econ_df = affordibility_df[affordibility_df["Area"] == country]


        
        cons = row.iloc[0]['2020']  

        fig_nutrition = px.bar(filtered_data, x='Nutrient', y='Median', title=f'Nutrient Consumption in {region} for {year_selected}')
        
        co2_little = pd.DataFrame({"Country": ["World", country], "2020": [47513.15/195, cons]})
        co2_fig=px.histogram(co2_little,x="Country",y="2020", histfunc='avg', title=f'CO2 release of {country} and worlds avg')



        if econ_df.empty:
            econ_child=html.Div(["",
                                 html.Br(),
                                 ""])
            
        else:
            cost = econ_df[econ_df["Item"] == "Cost of a healthy diet (PPP dollar per person per day)"]["Value"].values[0]
            percentage = econ_df[econ_df["Item"] == "Percentage of the population unable to afford a healthy diet (percent)"]["Value"].values[0]
            econ_child = html.Div([
                f"Cost of a healthy diet (PPP dollar per person per day) in {country} is {cost}$ in 2021.",
                html.Br(),
                f"Percentage of the population unable to afford a healthy diet (percent) in {country} is {percentage}% in 2021."
            ])
        if continent_row.empty:
            return fig_nutrition,co2_fig,no_update,econ_child
        else:
            subregion = continent_row["Unnamed: 1"].iloc[0]
            top_five_df=number_of_ing_df[number_of_ing_df["Region"]==subregion]
            top_fig=px.histogram(top_five_df, x="Count",y="Ingredient", title=f"Top 5 ingridients used in {subregion}")            
            return fig_nutrition,co2_fig,top_fig,econ_child
#start point again
# Helper function to populate country dropdown options
def get_country_options(df):
  countries = df['country name'].unique().tolist()
  # Create a list of dictionaries with labels and values
  return [{'label': c, 'value': c} for c in countries]

# Update country dropdown options based on chosen nutrient
@first.callback(
  Output(component_id='country-dropdown', component_property='options'),
  Input(component_id='nutrient-dropdown', component_property='value')
)
def update_country_options(nutrient):
  df = dataframes[nutrient]  # Access DataFrame based on nutrient
  return get_country_options(df)

# Callback to update the pie chart when selections change
@first.callback(
  Output(component_id='pie-chart', component_property='figure'),
  Input(component_id='nutrient-dropdown', component_property='value'),
  Input(component_id='country-dropdown', component_property='value')
)
def update_pie_chart(nutrient, selected_countries):
  # Filter to a maximum of 6 countries
  if len(selected_countries) > 6:
    selected_countries = selected_countries[:6]  # Truncate to max 6

  df_filtered = dataframes[nutrient]
  return create_pie_chart(df_filtered, nutrient, selected_countries)

# Callback to update the radar chart when selections change
@first.callback(
  Output(component_id='radar-chart', component_property='figure'),
  Input(component_id='nutrient-dropdown', component_property='value'),
  Input(component_id='country-dropdown', component_property='value')
)
def update_radar_chart(nutrient, selected_countries):
  # Extract regions from the pie chart data based on selected countries
  df_filtered = dataframes[nutrient]
  selected_regions = df_filtered[df_filtered['country name'].isin(selected_countries)]['Region'].unique()
  radar_df = radar_dataframes[nutrient]  # Get the corresponding radar DataFrame

  # Check if selected_regions is empty
  if len(selected_regions) == 0:
    return go.Figure()  # Return an empty figure if no regions are selected

  return create_radar_chart(radar_df, nutrient, selected_regions)
#end point again

if __name__ == '__main__':
    first.run(debug=True)