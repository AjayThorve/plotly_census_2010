# -*- coding: utf-8 -*-
import time
import os
import json
import gzip
import shutil
import requests
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
from pyproj import Transformer

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
from plotly.colors import sequential

from dask import delayed
from distributed import Client
from dask_cuda import LocalCUDACluster
import plotly.graph_objects as go
import pandas as pd
import cudf
import cupy

# Disable cupy memory pool so that cupy immediately releases GPU memory
cupy.cuda.set_allocator(None)

# Colors
bgcolor = "#191a1a"  # mapbox dark map land color
text_color = "#cfd8dc"  # Material blue-grey 100
mapbox_land_color = "#343332"


sex_categories = [0, 1] #male, female
sex_colors_list = ['lightblue', 'pink']
sex_colors = {cat: color for cat, color in zip(
    sex_categories, sex_colors_list
)}


bar_bgcolor = bgcolor
# Figure template
row_heights = [150, 440, 200]
template = {
    'layout': {
        'paper_bgcolor': bgcolor,
        'plot_bgcolor': bgcolor,
        'font': {'color': text_color},
        "margin": {"r": 0, "t": 30, "l": 0, "b": 20},
        'bargap': 0.05,
        'xaxis': {'showgrid': False, 'automargin': True},
        'yaxis': {'showgrid': True, 'automargin': True,
                  'gridwidth': 0.5, 'gridcolor': mapbox_land_color},
    }
}


# Load mapbox token from environment variable or file
token = os.getenv('MAPBOX_TOKEN')
if not token:
    token = open(".mapbox_token").read()

# geojson URL
puma_url = 'http://10.110.47.43:8080/census_geojson.json'

# Download geojson so we can read all of the puma codes we have
# response = requests.get(puma_url)
with open('./census_geojson.json') as f:
    puma_json = json.load(f)
valid_puma = {int(f['properties']['PUMA']) for f in puma_json["features"]}


# Names of float columns
float_columns = [
    'COW', 'SCHL', 'PUMA', 'JWAP', 'JWDP', 'PINCP',
    'ST', 'AGEP', 'SEX', 'WAOB', 'SOCP'
]

column_labels = {
    'PUMA': 'PUMA codes',
    'SEX': 'SEX',
    'COW': 'Class of Worker',
}

socp_mappings = {
"00": 'unknown',
"11": 'Mgmt',
"13": 'Bus-Fin',
"15": 'CS & Maths',
"17": 'Arch & Engg',
"19": 'Life, Phys & SSO',
"21": 'Comn & SSO',
"23": 'Legal',
"25": 'Edu & Lib',
"27": 'Arts, Design, Ent., Sports',
"29": 'Healthcare Prac',
"31": 'Healthcare Support',
"33": 'Protective Services',
"35": 'Food prep & Serving',
"37": 'Building & Maintenance',
"39": 'Personal Care and Service',
"41": 'Sales and Related',
"43": 'Office and Administrative Support',
"45": 'Farming, Fishing, and Forestry',
"47": 'Construction and Extraction',
"49": 'Maintenance, and Repair',
"51": 'Production',
"53": 'Transportation',
"55": 'Military',
"59": 'Unemployed'
}
socp_mappings_hover = {
"00": 'less than 16 years old/Never worked/last worked 5 years ago',
"11": 'Management Occupations',
"13": 'Business and Financial Operations Occupations',
"15": 'Computer and Mathematical Occupations',
"17": 'Architecture and Engineering Occupations',
"19": 'Life, Physical, and Social Science Occupations',
"21": 'Community and Social Service Occupations',
"23": 'Legal Occupations',
"25": 'Education, Training, and Library Occupations',
"27": 'Arts, Design, Entertainment, Sports, and Media Occupations',
"29": 'Healthcare Practitioners and Technical Occupations',
"31": 'Healthcare Support Occupations',
"33": 'Protective Service Occupations',
"35": 'Food Preparation and Serving Related Occupations',
"37": 'Building and Grounds Cleaning and Maintenance Occupations',
"39": 'Personal Care and Service Occupations',
"41": 'Sales and Related Occupations',
"43": 'Office and Administrative Support Occupations',
"45": 'Farming, Fishing, and Forestry Occupations',
"47": 'Construction and Extraction Occupations',
"49": 'Installation, Maintenance, and Repair Occupations',
"51": 'Production Occupations',
"53": 'Transportation and Material Moving Occupations',
"55": 'Military Services',
"59": 'Unemployed'
}


cow_mappings_hover = {
    0: "Not in universe (less than 16 years old/NILF who last worked more than 5 years ago or never worked)",
    1: "Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions",
    2: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
    3: "Local government employee (city, county, etc.)",
    4: "State government employee",
    5: "Federal government employee",
    6: "Self-employed in own not incorporated business, professional practice, or farm",
    7: "Self-employed in own incorporated business, professional practice or farm",
    8: "Working without pay in family business or farm",
    9: "Unemployed and last worked 5 years ago or earlier or never worked",
}

cow_mappings = {
    0: "never worked",
    1: "Emp private for-profit",
    2: "Emp private not-for-profit",
    3: "Local gov emp",
    4: "State gov emp",
    5: "Federal gov emp",
    6: "Self-emp in own not business",
    7: "Self-emp in own business",
    8: "family business or farm",
    9: "Unemployed",
}


education_mappings= {
"0": 'unknown',
'01': "No schooling completed",
'02': "Nursery school, preschool",
'03': "Kindergarten",
'04': "Grade 1",
'05': "Grade 2",
'06': "Grade 3",
'07': "Grade 4",
'08': "Grade 5",
'09': "Grade 6",
'10': "Grade 7",
'11': "Grade 8",
'12': "Grade 9",
'13': "Grade 10",
'14': "Grade 11",
'15': "12th grade - no diploma",
'16': "Regular high school diploma",
'17': "GED or alternative credential",
'18': "Some college, but less than 1 year",
'19': "1 or more years of college credit, no degree",
'20': "Associate's degree",
'21': "Bachelor's degree",
'22': "Master's degree",
'23': "Professional degree beyond a bachelor's degree",
'24': "Doctorate degree",
}

income_mappings = {
    -1 : 'unknown',
    0: '<25k',
    1: '25k - 50k',
    2: '50k - 75k',
    3: '75k - 100k',
    4: '100k - 125k',
    5: '125k - 150k',
    6: '150k - 175k',
    7: '175k - 200k',
    8: '200k - 225k',
    9: '225k - 250k',
    10: '250k - 1M',
    11: '1M+'
}

def load_dataset(path):
    """
    Args:
        path: Path to arrow file containing mortgage dataset

    Returns:
        pandas DataFrame
    """
    global total_sex_counts, data_center_3857, data_3857
    df = cudf.read_parquet(path)
    return df

# Build Dash app and initial layout
def blank_fig(height):
    """
    Build blank figure with the requested height
    Args:
        height: height of blank figure in pixels
    Returns:
        Figure dict
    """
    return {
        'data': [],
        'layout': {
            'height': height,
            'template': template,
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
        }
    }


app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.Div([
        html.H1(children=[
            'Census Data',
            html.A(
                html.Img(
                    src="assets/dash-logo.png",
                    style={'float': 'right', 'height': '50px', 'margin-right': '2%'}
                ), href="https://dash.plot.ly/"),
        ], style={'text-align': 'left'}),
    ]),
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.H4([
                    "Selected Polpulation",
                ], className="container_title"),
                dcc.Loading(
                    dcc.Graph(
                        id='indicator-graph',
                        figure=blank_fig(row_heights[0]),
                        config={'displayModeBar': False},
                    ),
                    style={'height': row_heights[0]},
                ),
                html.Div(children=[
                    html.Button(
                        "Clear All Selections", id='clear-all', className='reset-button'
                    ),
                ]),
            ], className='six columns pretty_container', id="indicator-div"),
            html.Div(children=[
                html.H4([
                    "Configuration",
                ], className="container_title"),
                html.Table([
                    html.Col(style={'width': '100px'}),
                    html.Col(),
                    html.Col(),
                    html.Tr([
                        html.Td(
                            html.Div("GPU"), className="config-label"
                        ),
                        html.Td(daq.DarkThemeProvider(daq.BooleanSwitch(
                            on=True,
                            color='#00cc96',
                            id='gpu-toggle',
                        ))),
                        html.Td(html.Button(
                            "Reset GPU", id='reset-gpu', style={'width': '100%'}
                        )),
                        html.Div(id='reset-gpu-complete', style={'display': 'hidden'})
                    ]),
                    html.Tr([
                        html.Td(html.Div("Color by"), className="config-label"),
                        html.Td(dcc.Dropdown(
                            id='aggregate-dropdown',
                            options=[
                                {'label': agg, 'value': agg}
                                for agg in ['count', 'mean', 'min', 'max']
                            ],
                            value='count',
                            searchable=False,
                            clearable=False,
                        )),
                        html.Td(dcc.Dropdown(
                            id='aggregate-col-dropdown',
                            value='PUMA',
                            searchable=False,
                            clearable=False,
                        )),
                    ]),
                    html.Tr([
                        html.Td(html.Div("Colormap"), className="config-label"),
                        html.Td(dcc.Dropdown(
                            id='colorscale-dropdown',
                            options=[
                                {'label': cs, 'value': cs}
                                for cs in ['Viridis', 'Cividis', 'Inferno', 'Magma', 'Plasma']
                            ],
                            value='Viridis',
                            searchable=False,
                            clearable=False,
                        )),
                        html.Td(dcc.Dropdown(
                            id='colorscale-transform-dropdown',
                            options=[{'label': t, 'value': t}
                                     for t in ['linear', 'sqrt', 'cbrt', 'log']],
                            value='linear',
                            searchable=False,
                            clearable=False,
                        )),
                    ]),
                    html.Tr([
                        html.Td(html.Div("Bin Count"), className="config-label"),
                        html.Td(dcc.Slider(
                            id='nbins-slider',
                            min=10,
                            max=40,
                            step=5,
                            value=20,
                            marks={m: str(m) for m in range(10, 41, 5)},
                            included=False,
                        ), colSpan=2),
                    ])
                ], style={'width': '100%', 'height': f'{row_heights[0] + 40}px'}),
            ], className='six columns pretty_container', id="config-div"),
        ]),
        html.Div(children=[
            html.H4([
                "US Population",
            ], className="container_title"),
            dcc.Graph(
                id='map-graph',
                figure=blank_fig(row_heights[1]),
                config={'displayModeBar': False},
            ),
            html.Button("Clear Selection", id='reset-map', className='reset-button'),
        ], className='twelve columns pretty_container',
            style={
                'width': '98%',
                'margin-right': '0',
            },
            id="map-div"
        ),
        html.Div(children=[
            html.H4([
                "Sex",
            ], className="container_title"),
            dcc.Graph(
                id='sex-radio',
                figure=blank_fig(row_heights[1]),
            ),
            html.Button("Clear Selection", id='reset-sex-radio', className='reset-button'),
        ], className='twelve columns pretty_container',
            style={
                'width': '98%',
                'margin-right': '0',
            },
            id="sex-radio-div"
        ),
    ]),
    html.Div(
        [
            html.H4('Acknowledgements', style={"margin-top": "0"}),
            dcc.Markdown('''\
 - Dashboard written in Python using the [Dash](https://dash.plot.ly/) web framework.
 - GPU accelerated provided by the [cudf](https://github.com/rapidsai/cudf) and
 [cupy](https://cupy.chainer.org/) libraries.
 - Base map layer is the ["dark" map style](https://www.mapbox.com/maps/light-dark/)
 provided by [mapbox](https://www.mapbox.com/).
'''),
        ],
        style={
            'width': '98%',
            'margin-right': '0',
            'padding': '10px',
        },
        className='twelve columns pretty_container',
    ),
])

# Register callbacks
@app.callback(
    [Output('aggregate-col-dropdown', 'options'),
     Output('aggregate-col-dropdown', 'disabled')],
    [Input('aggregate-dropdown', 'value')]
)
def update_agg_col_dropdown(agg):
    if agg == 'count':
        options = [{'label': 'NA',
                    'value': 'NA'}]
        disabled = True
    else:
        options = [{'label': v, 'value': k} for k, v in column_labels.items()]
        disabled = False
    return options, disabled


# Clear/reset button callbacks
@app.callback(
    Output('map-graph', 'selectedData'),
    [Input('reset-map', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_map(*args):
    return None

@app.callback(
    Output('sex-radio', 'selectedData'),
    [Input('reset-sex-radio', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_sex_hist_selections(*args):
    return None

# Query string helpers
def bar_selection_to_query(selection, column):
    """
    Compute pandas query expression string for selection callback data

    Args:
        selection: selectedData dictionary from Dash callback on a bar trace
        column: Name of the column that the selected bar chart is based on

    Returns:
        String containing a query expression compatible with DataFrame.query. This
        expression will filter the input DataFrame to contain only those rows that
        are contained in the selection.
    """
    point_inds = [p['pointIndex'] for p in selection['points']]
    xmin = min(point_inds) #bin_edges[min(point_inds)]
    xmax = max(point_inds) + 1 #bin_edges[max(point_inds) + 1]
    xmin_op = "<="
    xmax_op = "<="
    return f"{xmin} {xmin_op} {column} and {column} {xmax_op} {xmax}"


def build_query(selections, exclude=None):
    """
    Build pandas query expression string for cross-filtered plot

    Args:
        selections: Dictionary from column name to query expression
        exclude: If specified, column to exclude from combined expression

    Returns:
        String containing a query expression compatible with DataFrame.query.
    """
    other_selected = {sel for c, sel in selections.items() if c != exclude}
    if other_selected:
        return ' and '.join(other_selected)
    else:
        return None


# Plot functions
def build_colorscale(colorscale_name, transform):
    """
    Build plotly colorscale

    Args:
        colorscale_name: Name of a colorscale from the plotly.colors.sequential module
        transform: Transform to apply to colors scale. One of 'linear', 'sqrt', 'cbrt',
        or 'log'

    Returns:
        Plotly color scale list
    """
    colors = getattr(sequential, colorscale_name)
    if transform == "linear":
        scale_values = np.linspace(0, 1, len(colors))
    elif transform == "sqrt":
        scale_values = np.linspace(0, 1, len(colors)) ** 2
    elif transform == "cbrt":
        scale_values = np.linspace(0, 1, len(colors)) ** 3
    elif transform == "log":
        scale_values = (10 ** np.linspace(0, 1, len(colors)) - 1) / 9
    else:
        raise ValueError("Unexpected colorscale transform")
    return [(v, clr) for v, clr in zip(scale_values, colors)]


# Helper function to build figures
def build_sex_histogram(selected_sex_counts, selection_cleared, total_sex_counts):
    """
    Build horizontal histogram of radio counts
    """
    selectedpoints = False if selection_cleared else None
    hovertemplate = '%{x:,.0}<extra></extra>'

    fig = {'data': [
        {'type': 'bar',
            'x': total_sex_counts.tolist(),
            'y': total_sex_counts.to_array().tolist(),
            'marker': {'color': bar_bgcolor},
            'orientation': 'h',
            "selectedpoints": selectedpoints,
            'selected': {'marker': {'opacity': 1, 'color': bar_bgcolor}},
            'unselected': {'marker': {'opacity': 1, 'color': bar_bgcolor}},
            'showlegend': False,
            'hovertemplate': hovertemplate,
            },
        ], 
        'layout': {
            'barmode': 'overlay',
            'dragmode': 'select',
            'selectdirection': 'v',
            'clickmode': 'event+select',
            'selectionrevision': True,
            'height': 150,
            'margin': {'l': 10, 'r': 80, 't': 10, 'b': 10},
            'xaxis': {
                'type': 'log',
                'title': {'text': 'Count'},
                'range': [-1, np.log10(total_sex_counts.max() * 2)],
                'automargin': True,
            },
            'yaxis': {
                'type': 'category',
                'categoryorder': 'array',
                'categoryarray': sex_categories,
                'side': 'left',
                'automargin': True,
            },
    }}

    # Add selected bars in color
    fig['data'].append(
        {'type': 'bar',
            'x': selected_sex_counts.tolist(),
            'y': total_sex_counts.to_array().tolist(),
            'orientation': 'h',
            'marker': {'color': [sex_colors[cat] for cat in total_sex_counts.index]},
            "selectedpoints": selectedpoints,
            'unselected': {'marker': {'opacity': 0.2}},
            'hovertemplate': hovertemplate,
            'showlegend': False
        }
    )

    print(type(fig))
    return fig


def build_updated_figures(
        df_d, relayout_data, selected_sex,
        aggregate, aggregate_column, colorscale_name,
        transform
):
    """
    Build all figures for dashboard

    Args:
        df: pandas or cudf DataFrame
        relayout_data: selectedData for scattergeo
        selected_age_male: selectedData for age-male histogram
        selected_age_female: selectedData for age-female histogram
        selected_occupation: selectedData for occupation histogram
        selected_cow: selectedData for class of worker histogram
        selected_scatter_graph: selectedData for education-income scatter plot
        aggregate: Aggregate operation for choropleth (count, mean, etc.)
        aggregate_column: Aggregate column for choropleth
        colorscale_name: Colorscale name from plotly.colors.sequential
        colorscale_transform: Colorscale transformation ('linear', 'sqrt', 'cbrt', 'log')

    Returns:
        tuple of figures in the following order
        (choropleth, age_male_histogram, age_female_histogram,
        occupation_histogram, cow_histogram, scatter_graph,
        n_selected_indicator)
    """
    data_3857 = (
        [df_d['x'].min(), df_d['y'].min()],
        [df_d['x'].max(), df_d['y'].max()],
    )
    data_center_3857 = [[
        (data_3857[0][0] + data_3857[1][0]) / 2.0,
        (data_3857[0][1] + data_3857[1][1]) / 2.0,
    ]]
    # data_4326 = epsg_3857_to_4326(data_3857)
    # data_center_4326 = epsg_3857_to_4326(data_center_3857)

    total_sex_counts = df_d.sex.value_counts()
    print(relayout_data)

    coordinates_4326 = relayout_data and relayout_data.get('mapbox._derived', {}).get('coordinates', None)

    if coordinates_4326:
        lons, lats = zip(*coordinates_4326)
        lon0, lon1 = max(min(lons), data_3857[0][0]), min(max(lons), data_3857[1][0])
        lat0, lat1 = max(min(lats), data_3857[0][1]), min(max(lats), data_3857[1][1])
        coordinates_4326 = [
            [lon0, lat0],
            [lon1, lat1],
        ]
        coordinates_3857 = coordinates_4326
        # position = {}
        position = {
            'zoom': relayout_data.get('mapbox.zoom', None),
            'center': relayout_data.get('mapbox.center', None)
        }
    else:
        position = {
            'zoom': 0.5,
            'pitch': 0,
            'bearing': 0,
            'center': {'lon': data_center_3857[0][0], 'lat': data_center_3857[0][1]}
        }
        coordinates_3857 = data_3857
        coordinates_4326 = data_3857

    new_coordinates = [
        [coordinates_4326[0][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[0][1]],
        [coordinates_4326[0][0], coordinates_4326[0][1]],
    ]


    x_range, y_range = zip(*coordinates_4326)
    x0, x1 = x_range
    y0, y1 = y_range

    # Build query expressions
    query_expr_xy = f"(x >= {x0}) & (x <= {x1}) & (y >= {y0}) & (y <= {y1})"
    query_expr_range_sex_column_parts = []

    # Handle sex selection
    sex_col_slice = slice(None, None)
    if selected_sex:
        sex_columns = list(set(
                point['pointNumber'] for point in selected_sex['points']
            ))
        print(f"(sex in {sex_columns})")
        query_expr_range_sex_column_parts.append(
            f"(sex in {sex_columns})"
        )

    # Build dataframe containing rows that satisfy the range and created selections
    if query_expr_range_sex_column_parts:
        query_expr_range_sex_column = ' & '.join(query_expr_range_sex_column_parts)
        ddf_selected_range_sex = df_d.query(
            query_expr_range_sex_column
        )
    else:
        ddf_selected_range_sex = df_d

    # print(df_d)
    # print(query_expr_xy)
    # Build dataframe containing rows of towers within the map viewport
    df_d_t = df_d.query(query_expr_xy) if query_expr_xy else df_d

    # Build map figure
    # Create datashader aggregation of x/y data that satisfies the range and created
    # histogram selections
    cvs = ds.Canvas(
        plot_width=700,
        plot_height=400,
        x_range=x_range, y_range=y_range
    )
    agg = cvs.points(
        ddf_selected_range_sex, x='x', y='y', agg=ds.count('sex')
    )

    

    # Count the number of selected towers
    temp = agg.sum()
    temp.data = cupy.asnumpy(temp.data)
    n_selected = int(temp)

    # Build indicator figure
    n_selected_indicator = {
        'data': [{
            'type': 'indicator',
            'value': n_selected,
            'number': {
                'font': {
                    'color': '#263238'
                }
            }
        }],
        'layout': {
            'template': template,
            'height': 150,
            'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
        }
    }
    
    if n_selected == 0:
        # Nothing to display
        lat = [None]
        lon = [None]
        customdata = [None]
        marker = {}
        layers = []
    elif n_selected < 5000:
        # Display each individual point using a scattermapbox trace. This way we can
        # give each individual point a tooltip
        ddf_gpu_small_expr = ' & '.join(
            [query_expr_xy] + [f'(sex in {selected_sex_categories})'] +
            query_expr_range_created_parts
        )
        print(ddf_gpu_small_expr)
        ddf_gpu_small = df_d.query(ddf_gpu_small_expr).to_pandas()
        print("querying done")

        x, y, sex = (
            ddf_gpu_small.x, ddf_gpu_small.y, ddf_gpu_small.sex
        )

        # Format creation date column for tooltip
        created = pd.to_datetime(created.tolist()).strftime('%x')

        # Build colorscale to give scattermapbox points the appropriate color
        sex_colorscale = [
            [v, sex_colors[cat]] for v, cat in zip(
                np.linspace(0, 1, len(sex.unique().tolist())), sex.unique().tolist()
            )
        ]

        # Build array of the integer category codes to use as the numeric color array
        # for the scattermapbox trace
        sex_codes = sex.unique().tolist()

        # Build marker properties dict
        marker = {
            'color': sex_codes,
            'colorscale': sex_colorscale,
            'cmin': 0,
            'cmax': 3,
            'size': 5,
            'opacity': 0.6,
        }

        customdata = list(zip(
            sex.astype(str)
        ))
        layers = []
    else:
        # Shade aggregation into an image that we can add to the map as a mapbox
        # image layer
        img = tf.shade(agg, color_key=sex_colors, min_alpha=100).to_pil()

        # Resize image to map size to reduce image blurring on zoom.
        img = img.resize((1920, 1080))

        # Add image as mapbox image layer. Note that as of version 4.4, plotly will
        # automatically convert the PIL image object into a base64 encoded png string
        layers = [
            {
                "sourcetype": "image",
                "source": img,
                "coordinates": new_coordinates
            }
        ]

        # Do not display any mapbox markers
        lat = [None]
        lon = [None]
        customdata = [None]
        marker = {}

    # Build map figure
    map_graph = {
        'data': [{
            'type': 'scattermapbox',
            'lat': lat, 'lon': lon,
            'customdata': customdata,
            'marker': marker,
            'hovertemplate': (
                "sex: %{customdata[0]}<br>"
                "<extra></extra>"
            )
        }],
        'layout': {
            'template': template,
            'uirevision': True,
            'mapbox': {
                'style': "light",
                'accesstoken': token,
                'layers': layers,
            },
            'margin': {"r": 0, "t": 0, "l": 0, "b": 0},
            'height': 500,
            'shapes': [{
                'type': 'rect',
                'xref': 'paper',
                'yref': 'paper',
                'x0': 0,
                'y0': 0,
                'x1': 1,
                'y1': 1,
                'line': {
                    'width': 2,
                    'color': '#B0BEC5',
                }
            }]
        },
    }

    map_graph['layout']['mapbox'].update(position)
    selected_sex_counts = df_d_t.sex.value_counts()

    sex_histogram = build_sex_histogram(
        selected_sex_counts, selected_sex is None, total_sex_counts
    )

    return (n_selected_indicator, map_graph, sex_histogram)


def register_update_plots_callback(client):
    """
    Register Dash callback that updates all plots in response to selection events
    Args:
        df_d: Dask.delayed pandas or cudf DataFrame
    """
    @app.callback(
        [Output('indicator-graph', 'figure'), Output('map-graph', 'figure'),
         Output('sex-radio', 'figure')
         ],
        [   Input('map-graph', 'selectedData'), Input('sex-radio', 'selectedData'),
            Input('aggregate-dropdown', 'value'), Input('aggregate-col-dropdown', 'value'),
            Input('colorscale-dropdown', 'value'), Input('colorscale-transform-dropdown', 'value'),
            Input('gpu-toggle', 'on')
        ]
    )
    def update_plots(
            selected_map, selected_sex,
            aggregate, aggregate_column, colorscale_name, transform, gpu_enabled
    ):
        t0 = time.time()
        # Get delayed dataset from client
        if gpu_enabled:
            df_d = client.get_dataset('c_df_d')
        else:
            df_d = client.get_dataset('pd_df_d')

        print('udpated')
        figures_d = delayed(build_updated_figures)(
            df_d, selected_map, selected_sex,
            aggregate, aggregate_column, colorscale_name,
            transform)

        figures = figures_d.compute()
        (n_selected_indicator, map_graph, sex_histogram) = figures

        print(f"Update time: {time.time() - t0}")
        return (
            n_selected_indicator, map_graph, sex_histogram
        )


def publish_dataset_to_cluster():

    data_path = "/home/ajay/new_dev/plotly/census_large/plotly_epsg4857.parquet/*"

    # Note: The creation of a Dask LocalCluster must happen inside the `__main__` block,
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="1")
    client = Client(cluster)
    print(f"Dask status: {cluster.dashboard_link}")

    # Load dataset and persist dataset on cluster
    def load_and_publish_dataset():
        # cudf DataFrame
        c_df_d = delayed(load_dataset)(data_path).persist()
        # pandas DataFrame
        pd_df_d = delayed(c_df_d.to_pandas)().persist()

        # Unpublish datasets if present
        for ds_name in ['pd_df_d', 'c_df_d']:
            if ds_name in client.datasets:
                client.unpublish_dataset(ds_name)

        # Publish datasets to the cluster
        client.publish_dataset(pd_df_d=pd_df_d)
        client.publish_dataset(c_df_d=c_df_d)

    load_and_publish_dataset()

    # Precompute field bounds
    c_df_d = client.get_dataset('c_df_d')

    # Define callback to restart cluster and reload datasets
    @app.callback(
        Output('reset-gpu-complete', 'children'),
        [Input('reset-gpu', 'n_clicks')]
    )
    def restart_cluster(n_clicks):
        if n_clicks:
            print("Restarting LocalCUDACluster")
            client.unpublish_dataset('pd_df_d')
            client.unpublish_dataset('c_df_d')
            client.restart()
            load_and_publish_dataset()

    # Register top-level callback that updates plots
    register_update_plots_callback(client)


def server():
    # gunicorn entry point when called with `gunicorn 'app:server()'`
    publish_dataset_to_cluster()
    return app.server


if __name__ == '__main__':
    # development entry point
    publish_dataset_to_cluster()

    # Launch dashboard
    app.run_server(debug=False, dev_tools_silence_routes_logging=True)