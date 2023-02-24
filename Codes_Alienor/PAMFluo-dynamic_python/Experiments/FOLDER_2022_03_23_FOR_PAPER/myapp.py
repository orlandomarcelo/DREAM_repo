import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import json
import random
import binascii
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import itertools


if False: 
    c = {}
    #colors = generate_col()
    colors = list(itertools.product(list(range(30, 256)), repeat=3))
    random.seed(8)
    random.shuffle(colors)
    for i in range(1000):
        c[i] = "rgb(%d,%d,%d)"%(colors[i][0], colors[i][1], colors[i][2])
    with open('plotly_data/color_map.json', 'w') as fp:
        json.dump(c, fp)    
        
        
def get_idxs(id_list, sel_classes): #select experiment
   idxs = []

   for i,c in enumerate(sel_classes):  
      nidxs=np.where(id_list==c)[0]
      idxs=np.concatenate([idxs,nidxs])
   return idxs

def create_trace(X_sel, idexp, idx):
    title = '<b>{}</b><b>{}</b><br>{}</br>'.format(class_match[idexp], idexp, idx) #titre

    fig_trace = px.scatter(np.array(X_sel))
    
    fig_trace.update_traces(mode='lines')
    fig_trace.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig_trace.update_layout(height=200, 
                            width = 400, 
                            margin={'l': 10, 'b': 10, 'r': 10, 't': 10},
                            plot_bgcolor='rgba(255,255,255,1)'
)
    fig_trace.update_xaxes(showgrid=False)
    fig_trace.update_yaxes(showgrid=False)
    
    return fig_trace

def create_fig(data, method, x, y):
    X0 = X[data]
    Z0 = Z[method][data]
    label_list0 = label_list[data]
    id_list0 = id_list[data]
    algae_list0 = algae_list[data]

    class_color = np.zeros(label_list0.shape).astype(str)
    for i, c in enumerate(np.unique(label_list0)):
        class_color[label_list0 == c] =colmap[i]
    slash = np.array(len(label_list0)*["/"])
    clab = np.char.add(label_list0.astype(str), slash)
    idlist = algae_list0

    name_no_id = np.char.add(clab, id_list0.astype(str))
    name_no_id = np.char.add(name_no_id, slash)
    names_with_id = np.char.add(name_no_id, idlist.astype(str))
    fig = px.scatter(x = Z0[:,x], y = Z0[:,y], hover_name=names_with_id, color=class_color)#id_list0.astype(str))#, color_discrete_sequence=colmap)
    fig.update_traces(marker_size = 6)
    fig.update_layout(height = 500, 
                      width=550, 
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_range=lims[method]["%d,%d"%(x,y)][0],
                      yaxis_range=lims[method]["%d,%d"%(x,y)][1],
                      plot_bgcolor='rgba(255,255,255,1)'

                      )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def create_image(X_sel, idexp, idx):
    image = imref_list[int(idexp)].T
    image = image/image.max()*255
    image = image//1
    algae = mask_list[int(idexp)] == int(idx)
    algae = algae.T
    image[algae] = 0
    
    image = np.array([algae.astype(int)*255, image, image]).T
    
    fig = px.imshow(image)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig  
    
    

data_folder = "plotly_data/"

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "FluoDL"

X = np.load(data_folder + 'traces_list.npy') # pulses
Z = {}
method_list = ['combine0', 'combine1', 'combine2', 'combine3', 'combine4', "pulses", "dict", "simple"]
for method in method_list:
    Z[method] = np.load(data_folder + method + "_array_proj.npy") # projections des dictionnaires 3D résultat LDA
    
label_list = np.load(data_folder + 'label_list.npy')
algae_list = np.load(data_folder + 'algae_list.npy')
imref_list = np.load(data_folder + 'imref_list.npy')
mask_list = np.load(data_folder + 'mask_list.npy')

class_match = json.load(open(data_folder + "class_match.json"))
colmap = json.load(open(data_folder + "color_map.json"))# colormap 
colmap = {int(k):v for k,v in colmap.items()}
id_list = np.load(data_folder + 'id_list.npy')
exp_array = np.load(data_folder + 'exp_list.npy')

lims = {}
for method in method_list:
    lims[method] = json.load(open(data_folder + method + "_array_ax_lim.json"))



app.layout = html.Div(
    [
    
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.B("Experimental conditions"),
                        dcc.Dropdown(options = [str(i) + '_' + str(j) + '_' + str(k) for i in range(12) for j in range(10) for k in range(4)],
                                     value = ['0_0_3', '0_1_3', '1_0_3'],
                                     multi=True, 
                                     id="exp_cond")
                    ], width=3
                    
                ), 
                
                dbc.Col(
                    [
                        html.B("Features"),
                        dcc.RadioItems(options = method_list, 
                                       value = 'combine0', 
                                       id = 'method', 
                                       labelStyle = dict(display='block')) 
                    ], width=3
                ), 
                
                dbc.Col(
                    [
                        html.B("2D space"),
                        dcc.RadioItems(options = ["qT, qE", "qT, qI", "qE, qI"], 
                                       value = "qT, qE", 
                                       id = 'representation', 
                                       labelStyle = dict(display='block'))
                    ], width=3
                )
                                
            ]
        ),

    
        dbc.Row(
            [
                #html.Label('Experiment details '),
                dbc.Col(dcc.Checklist(options = ['0_0_0/activation_4H_ENS_wt4/12', '0_0_0/activation_4H_ENS_stt7/0'],
                              value = ["0_0_3/activation_4H_ENS_stt7/3",
                                       "0_1_3/activation_4H_ENS_stt7/7",
                                       "1_0_3/activation_4H_ENS_wt4/15",
                                       "0_0_0/activation_4H_ENS_stt7/0",
                                       "1_1_3/HL_chunks_WT4_backup/61",
                                       "1_2_3/HL_chunks_WT4_backup/65",
                                       "1_3_3/HL_chunks_WT4_backup/69"], 
                              id = "exp_desc",
                              labelStyle = dict(display='block'))), 
                
                dbc.Col(
                    dcc.Graph(id="3dscatter", hoverData={'points': [{'hovertext': "0_0_0/3/1"}]}), width=4),
                dbc.Col(
                    [
                        dbc.Row(dcc.Graph(id='trace')), 
                        dbc.Row(dcc.Graph(id='segment'))
                    ]
                    ),
            ]
        ), 
        
        dbc.Row(
            [
                dcc.Store(id='exp_details'), 
                dcc.Store(id='intermediate-value'), 
                dcc.Store(id='selected-algae')
            ]
        )
    ]
)
    
                       


@app.callback(Output('exp_details', 'data'), 
              [Input('exp_cond', 'value')]) #appelé chaque fois que hoverdata change
def selected_data(value):
    labels = []
    idx = []
    desc = []
    for v in value: 
        u = np.unique(id_list[label_list==v])
        idx.append(u)
        loco = []
        for i in u: #FOLDER_2022_01_
            loco.append(class_match[i.astype(str)][18:])
        desc.append(loco) #0
        labels.append([v]*len(u)) #0_0_0
    idx = np.array(np.concatenate(idx))
    labels = np.array(np.concatenate(labels))
    desc = np.array(np.concatenate(desc))
    return np.array([labels, desc, idx])


@app.callback(Output('exp_desc', 'options'), 
              [Input('exp_details', 'data')]) #appelé chaque fois que hoverdata change
def selected_data(data):
    labels, desc, idx = data
    slash = np.array(["/"]*len(idx))
    out = np.char.add(labels, slash)
    out= np.char.add(out, np.array(desc))
    out = np.char.add(out, slash)
    out = np.char.add(out, idx)
    return out

@app.callback(Output('intermediate-value', 'data'), 
              [Input('exp_desc', 'value'), 
               Input("method", "value")]) #appelé chaque fois que hoverdata change
def selected_data(value, method):
    selected = []
    for v in value:
        idx = v.split("/")[-1]
        selected.append(idx)
    idxs = get_idxs(id_list, np.array(selected).astype(int))
    idxs = np.array(idxs, dtype=int)
    Z0 = Z[method][idxs]
    select = abs(Z0 - np.mean(Z0, axis = 0)) < 4 * np.std(Z0, axis = 0)
    select = select.min(axis = 1)    
    idxs = idxs[select]
    return idxs

@app.callback(
    Output('3dscatter', 'figure'),
    [Input('intermediate-value', 'data'),
     Input("method", "value"), 
    Input('representation', 'value')]) #appelé chaque fois que hoverdata change
def gen_fig(data, method, representation): #figure
    x, y = representation.split(', ')
    equiv = {"qT":0, "qE": 1, "qI": 2}
    return create_fig(data, method, equiv[x], equiv[y])

@app.callback(
    Output('selected-algae', 'data'),
    [Input('3dscatter', 'hoverData'), 
     Input('intermediate-value', 'data')]) #appelé chaque fois que hoverdata change
def selected_algae(hoverData, idxs):
    name = hoverData['points'][0]['hovertext'].split("/")
    idx = int(name[-1])#recupere indice de la cellule
    idexp = name[-2]
    id_list0 = np.array(id_list[idxs])
    algae_list0 = np.array(algae_list[idxs])
    u = (id_list0 == int(idexp))
    v = (algae_list0 == int(idx))
    X_sel = np.array(idxs)[u*v]
    return [X_sel, idexp, idx]

@app.callback(
    Output('trace', 'figure'),
    Input('selected-algae', 'data'))
def update_trace(data):
    trace = X[data[0]][0]
    idexp = data[1]
    idx = data[2]
    return create_trace(trace, idexp, idx)

@app.callback(
    Output('segment', 'figure'),
    Input('selected-algae', 'data'))
def update_trace(data):
    trace = X[data[0]][0]
    idexp = data[1]
    idx = data[2]
    return create_image(trace, idexp, idx)


if __name__ == '__main__':
    app.run_server(debug=True)