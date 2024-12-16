import json
import re
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import pandas as pd
from networkx import jaccard_coefficient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.lite.python.util import is_frozen_graph



def load_csv_data(path):
   G=nx.DiGraph()
   df = pd.read_csv(path, header=None)
   for index, row in df.iterrows():
        from_node,to_node,attitude=row[0],row[1],row[2]
        G.add_edge(from_node,to_node,label=attitude)

   return G

def count_features(G):
   Node_feature_list=[]
   for node in G.nodes:
       D_in=G.in_degree(node)
       D_out=G.out_degree(node)
       In_label = [data['label'] for _, _, data in G.in_edges(node, data=True)]
       Out_label = [data['label'] for _, _, data in G.out_edges(node, data=True)]
       In_label_pos=0
       In_label_neg=0
       Out_label_pos=0
       Out_label_neg=0
       for label in In_label:
            if label ==1:
               In_label_pos+=1
            else:
               In_label_neg+=1
       for label in Out_label:
            if label ==1:
               Out_label_pos+=1
            else:
               Out_label_neg+=1
       Node_feature_list.append({str(node):{'D_in':D_in,'D_out':D_out,'In_label_pos':In_label_pos,'In_label_neg':In_label_neg,
                                       'Out_label_pos':Out_label_pos,'Out_label_neg':Out_label_neg}})
   return Node_feature_list

def calculate_embeddedness(graph,u,v):
   if not graph.has_edge(u, v):
      return 0

   neighbors_u = set(graph.neighbors(u))
   neighbors_v = set(graph.neighbors(v))
   common_neighbors = neighbors_u & neighbors_v

   return len(common_neighbors)

def jaccard_similarity(graph,u,v):
    neighbors_u = set(graph.successors(u)) | set(graph.predecessors(u))
    neighbors_v = set(graph.successors(v)) | set(graph.predecessors(v))

    intersection = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v


    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

# degree features
def create_data(graph,edges):
    result_list= []
    D_in_list=[]
    D_out_list=[]
    In_label_pos_list=[]
    In_label_neg_list=[]
    Out_label_pos_list=[]
    Out_label_neg_list=[]
    embed_list=[]
    for u,v,data in edges:
        embed=calculate_embeddedness(graph,u,v)
        D_in= graph.in_degree(v)
        D_out = graph.out_degree(u)
        In_label = [data['label'] for _, _, data in graph.in_edges(v, data=True)]
        Out_label = [data['label'] for _, _, data in graph.out_edges(u, data=True)]
        In_label_pos=0
        In_label_neg=0
        Out_label_pos=0
        Out_label_neg=0
        for label in In_label:
            # print(type(label))
            if label == '1':
                In_label_pos += 1
            if label == '-1':
                In_label_neg += 1
        for label in Out_label:
            if label == '1':
                Out_label_pos += 1
            if label == '-1':
                Out_label_neg += 1
        D_in_list.append(D_in)
        D_out_list.append(D_out)
        In_label_pos_list.append(In_label_pos)
        In_label_neg_list.append(In_label_neg)
        Out_label_pos_list.append(Out_label_pos)
        Out_label_neg_list.append(Out_label_neg)
        embed_list.append(embed)
        result_list.append(data['label'])
    feature_Dict={'D_in':D_in_list,'D_out':D_out_list,'In_label_pos':In_label_pos_list,'In_label_neg':In_label_neg_list,
                                       'Out_label_pos':Out_label_pos_list,'Out_label_neg':Out_label_neg_list,'embed_list':embed_list}
    result_Dict={'result':result_list}
    data = pd.DataFrame(feature_Dict)
    result=pd.DataFrame(result_Dict)

    return data,result


# reg model
def creat_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model = LogisticRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("LogisticRegression_accuracy:", accuracy)
    return accuracy


# tree model
def creat_tree_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model  = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)

    print("tree_accuracy:", accuracy)
    return accuracy

# random forest
def creat_forest_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model  = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)

    print("forest_accuracy:", accuracy)
    return accuracy


# triangle feature
def creat_triangle(graph,edges):
    # (w → u, w → v): Positive - Positive
    # (w → u, w ← v): Positive - Positive
    # (w → u, w → v): Positive - Negative
    # (w → u, w ← v): Positive - Negative
    # (w ← u, w → v): Positive - Positive
    # (w ← u, w ← v): Positive - Positive
    # (w ← u, w → v): Positive - Negative
    # (w ← u, w ← v): Positive - Negative
    # (w → u, w → v): Negative - Positive

    # (w → u, w ← v): Negative - Positive

    # (w → u, w → v): Negative - Negative

    # (w → u, w ← v): Negative - Negative

    # (w ← u, w → v): Negative - Positive

    # (w ← u, w ← v): Negative - Positive

    # (w ← u, w → v): Negative - Negative

    # (w ← u, w ← v): Negative - Negative
    triangle_feature=[]
    result_list = []

    for u,v,data in edges:
        wu_wv_pp = 0
        wu_vw_pp = 0
        wu_wv_pn = 0
        wu_vw_pn = 0
        uw_wv_pp = 0
        uw_vw_pp = 0
        uw_wv_pn = 0
        uw_vw_pn = 0
        wu_wv_np = 0
        wu_vw_np = 0
        wu_wv_nn = 0
        wu_vw_nn = 0
        uw_wv_np = 0
        uw_vw_np = 0
        uw_wv_nn = 0
        uw_vw_nn = 0

        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        triangle_nodes = neighbors_u & neighbors_v
        for w in triangle_nodes:
            if graph.has_edge(w, u) and graph.has_edge(w, v):
                if graph.get_edge_data(w,u)['label'] == '1' and graph.get_edge_data(w,v)['label'] == '1':
                    wu_wv_pp +=1
                if graph.get_edge_data(w, u)['label'] == '1' and graph.get_edge_data(w, v)['label'] == '-1':
                    wu_wv_pn +=1
                if graph.get_edge_data(w, u)['label'] == '-1' and graph.get_edge_data(w, v)['label'] == '1':
                    wu_wv_np +=1
                if graph.get_edge_data(w, u)['label'] == '-1' and graph.get_edge_data(w, v)['label'] == '-1':
                    wu_wv_nn +=1
            if graph.has_edge(w, u) and graph.has_edge(v, w):
                if graph.get_edge_data(w,u)['label'] == '1' and graph.get_edge_data(v,w)['label'] == '1':
                    wu_vw_pp +=1
                if graph.get_edge_data(w, u)['label'] == '1' and graph.get_edge_data(v,w)['label'] == '-1':
                    wu_vw_pn +=1
                if graph.get_edge_data(w, u)['label'] == '-1' and graph.get_edge_data(v,w)['label'] == '1':
                    wu_vw_np +=1
                if graph.get_edge_data(w, u)['label'] == '-1' and graph.get_edge_data(v,w)['label'] == '-1':
                    wu_vw_nn+=1
            if graph.has_edge(u, w) and graph.has_edge(w, v):
                if graph.get_edge_data(u,w)['label'] == '1' and graph.get_edge_data(w,v)['label'] == '1':
                    uw_wv_pp +=1
                if graph.get_edge_data(u,w)['label'] == '1' and graph.get_edge_data(w, v)['label'] == '-1':
                    uw_wv_pn +=1
                if graph.get_edge_data(u,w)['label'] == '-1' and graph.get_edge_data(w, v)['label'] == '1':
                    uw_wv_np +=1
                if graph.get_edge_data(u,w)['label'] == '-1' and graph.get_edge_data(w, v)['label'] == '-1':
                    uw_wv_nn +=1
            if graph.has_edge(u, w) and graph.has_edge(v,w):
                if graph.get_edge_data(u,w)['label'] == '1' and graph.get_edge_data(v,w)['label'] == '1':
                    uw_vw_pp +=1
                if graph.get_edge_data(u,w)['label'] == '1' and graph.get_edge_data(v,w)['label'] == '-1':
                    uw_vw_pn +=1
                if graph.get_edge_data(u,w)['label'] == '-1' and graph.get_edge_data(v,w)['label'] == '1':
                    uw_vw_np +=1
                if graph.get_edge_data(u,w)['label'] == '-1' and graph.get_edge_data(v,w)['label'] == '-1':
                    uw_vw_nn +=1
        result_list.append(data['label'])
        triangle_feature.append([wu_wv_pp,wu_wv_pn,wu_wv_np,wu_wv_nn,wu_vw_pp,wu_vw_pn,wu_vw_np,wu_vw_nn,uw_wv_pp,uw_wv_pn,uw_wv_np,uw_wv_nn
                                 ,uw_vw_pp,uw_vw_pn,uw_vw_np,uw_vw_nn])

    triangle_Dict={'triangle_Dict':triangle_feature}
    result_Dict={'result':result_list}
    result=pd.DataFrame(result_Dict)
    triangle_Data=pd.DataFrame(triangle_Dict)

    return triangle_Data,result


# use jaccard accuracy
def similarity_calculate(graph,edges):

    result_list = []
    test_list=[]

    for u, v, data in edges:
        nodelist = []
        simlist=[]
        valid_common_neighbors =  set(graph.predecessors(v))


        for x in valid_common_neighbors:
            if x!=u:
                sim= jaccard_similarity(graph,x,u)
                nodelist.append(x)
                simlist.append(sim)
        if len(simlist)>0 and max(simlist)>0.5:
            node_with_max_sim = nodelist[simlist.index(max(simlist))]
            if node_with_max_sim not in valid_common_neighbors:
                print("000000")
                print(node_with_max_sim)
            edge_data = graph.get_edge_data(node_with_max_sim, v)
            if edge_data is None:
                print(f"No edge exists between {node_with_max_sim} and {v}")
            elif 'label' not in edge_data:
                print(f"Edge between {node_with_max_sim} and {v} has no 'label' attribute")
            test_list.append(edge_data['label'])
            # test_list.append(graph.get_edge_data(nodelist[simlist.index(max(simlist))],v)['label'])
        else:
            test_list.append(0)
        result_list.append(data['label'])
    print(len(test_list))
    print(len(result_list))
    correct = sum(1 for p, t in zip(test_list, result_list) if p == t)
    non_zero_count = len(list(filter(lambda x: x != 0, test_list)))
    print(non_zero_count)
    print(correct)
    accuracy = correct/non_zero_count

    return accuracy












if __name__ == "__main__":

  # load epinon data
  graph=load_csv_data('simplified_soc-sign-epinions.csv')
  edges=graph.edges(data=True)

  # load  wiki data
  graph_wiki=load_csv_data('simplified_wiki.csv')
  graph_edges=graph_wiki.edges(data=True)

  # load bitcoinalpha data
  bitgraph = load_csv_data('simplified-soc-sign-bitcoinalpha.csv')
  bitedges = bitgraph.edges(data=True)

  # load bitcoinotc data
  bitcointcgraph = load_csv_data('simplified-soc-sign-bitcoinotc.csv')
  bitcointcedges = bitcointcgraph.edges(data=True)

  # load Slashdot data
  slashgraph=load_csv_data('simplified_soc-sign-Slashdot090221.csv')
  slashedges = slashgraph.edges(data=True)



  #  different models compare between slashdot,epinon,wikipedia

  # epinon
  data,result=create_data(graph,edges)
  epinon=[]
  epinon.append(creat_model(data,result))
  epinon_result = [creat_model(data, result), creat_tree_model(data, result), creat_forest_model(data, result),similarity_calculate(graph,edges)]

  # slashdot

  data,result=create_data(slashgraph,slashedges)
  data.to_csv('slashdot_data.csv', index=False)
  result.to_csv('slashdot_result.csv', index=False)
  data = pd.read_csv('slashdot_data.csv')
  result = pd.read_csv('slashdot_result.csv')
  slashdot_result = []
  slashdot_result=[creat_model(data,result),creat_tree_model(data,result),creat_forest_model(data,result),similarity_calculate(slashgraph,slashedges)]

  # wikipedia
  wiki_result=[]
  data, result = create_data(graph_wiki, graph_edges)
  wiki_result.append(creat_model(data,result))

  wiki_result=[creat_model(data,result),creat_tree_model(data,result),creat_forest_model(data,result),similarity_calculate(graph_wiki,graph_edges)]
#
#
# #   plot
  models = ['RegModel', 'Tree Model', 'Forest Model','Jaccard similarity']
  width = 0.25
  x = np.arange(len(models))


  fig, ax = plt.subplots(figsize=(10, 6))


  ax.bar(x - width, epinon_result, width, label='Epinions')
  ax.bar(x, wiki_result, width, label='Wikipedia')
  ax.bar(x + width, slashdot_result, width, label='Slashdot')


  ax.set_xlabel('Models')

  ax.set_ylabel('Accuracy')
  ax.set_title('Accuracy Comparison for Different Models and Datasets')


  ax.set_xticks(x)
  ax.set_xticklabels(models)


  ax.legend()


  plt.tight_layout()
  plt.show()


# get triangle result

# epinon
  epinon=[]
  data,result=creat_triangle(graph,edges)
  data = data['triangle_Dict'].apply(pd.Series)
  epinon.append(creat_model(data,result))

# wiki
  wiki_result=[]
  data, result = creat_triangle(graph_wiki, graph_edges)
  data = data['triangle_Dict'].apply(pd.Series)
  wiki_result.append(creat_model(data,result))

# slashdot
  slashdot_result = []
  data, result = creat_triangle(slashgraph, slashedges)
  data = data['triangle_Dict'].apply(pd.Series)
  slashdot_result.append(creat_model(data,result))










# compare between 5 different dataset

    # epinon
  data,result=create_data(graph,edges)
  epinon=[]
  epinon.append(creat_model(data,result))

    # slashdot
  ata,result=create_data(slashgraph,slashedges)
  data.to_csv('slashdot_data.csv', index=False)
  result.to_csv('slashdot_result.csv', index=False)
  data = pd.read_csv('slashdot_data.csv')
  result = pd.read_csv('slashdot_result.csv')
  slashdot_result = []

  slashdot_result.append(creat_model(data,result))

    #wiki
  wiki_result=[]
  data, result = create_data(graph_wiki, graph_edges)
  wiki_result.append(creat_model(data,result))

    # bitcoinalpha
  data,result=create_data(bitgraph,bitedges)
  bit_res=[]
  bit_res.append(creat_model(data,result))

    # bitcointc

  data,result=create_data(bitcointcgraph,bitcointcedges)
  bitcoin_res=[]
  bitcoin_res.append(creat_model(data,result))

  categories = ['bitcointc', 'bitcoinalpha', 'wikipedia', 'slashdot', 'epinon']
  accuracies = [bitcoin_res[0], bit_res[0], wiki_result[0], slashdot_result[0], epinon[0]]


  x = np.arange(len(categories))
  width = 0.6


  fig, ax = plt.subplots(figsize=(8, 6))
  bars = ax.bar(x, accuracies, width, color='skyblue')


  for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')


  ax.set_xlabel('Categories', fontsize=12)
  ax.set_ylabel('Accuracy', fontsize=12)
  ax.set_title('Accuracy Comparison Across dataset', fontsize=14)
  ax.set_xticks(x)
  ax.set_xticklabels(categories)
  ax.set_ylim(0, 1)


  plt.tight_layout()
  plt.show()