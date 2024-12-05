import json

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


def load_data(path):
   G=nx.DiGraph()
   with open(path, 'r') as f:
      for _ in range(4):
         next(f)
      for line in f:
         from_node,to_node,attitude=line.split()
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
       # print(D_out)
       # print(Out_label)
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

def create_data(graph,edges):
    feature_list=[]
    result_list= []
    D_in_list=[]
    D_out_list=[]
    In_label_pos_list=[]
    In_label_neg_list=[]
    Out_label_pos_list=[]
    Out_label_neg_list=[]
    embed_list=[]
    # features=count_features(graph)
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
            else:
                In_label_neg += 1
        for label in Out_label:
            if label == '1':
                Out_label_pos += 1
            else:
                Out_label_neg += 1

        # for feature in features:
        #     # print(str(next(iter(feature))),edge[0])
        #     if str(next(iter(feature)))== edge[0]:
        #         D_in=feature[edge[0]]['D_in']
        #         In_label_pos=feature[edge[0]]['In_label_pos']
        #         In_label_neg=feature[edge[0]]['In_label_neg']
        #     if str(next(iter(feature)))== edge[1]:
        #         D_out=feature[edge[1]]['D_out']
        #         Out_label_pos=feature[edge[1]]['Out_label_pos']
        #         Out_label_neg=feature[edge[1]]['Out_label_neg']
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


def creat_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model = LogisticRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("LogisticRegression_accuracy:", accuracy)

def creat_svm_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model  = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("svm_accuracy:", accuracy)

def creat_tree_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model  = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("tree_accuracy:", accuracy)

def creat_forest_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)


    model  = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("forest_accuracy:", accuracy)

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
    # triangle_Dict={'wu_wv_pp':wu_wv_pp,'wu_wv_pn':wu_wv_pn,'wu_wv_np':wu_wv_np,'wu_wv_nn':wu_wv_nn,'wu_vw_pp':wu_vw_pp,
    #                'wu_vw_pn':wu_vw_pn,'wu_vw_np':wu_vw_np,'wu_vw_nn':wu_vw_nn,'uw_wv_pp':uw_wv_pp,'uw_wv_pn':uw_wv_pn,
    #               'uw_wv_np':uw_wv_np,'uw_wv_nn':uw_wv_nn,'uw_vw_pp':uw_vw_pp,'uw_vw_pn':uw_vw_pn,'uw_vw_np':uw_vw_np,
    #                'uw_vw_nn':uw_vw_nn}

    triangle_Dict={'triangle_Dict':triangle_feature}
    result_Dict={'result':result_list}
    result=pd.DataFrame(result_Dict)
    triangle_Data=pd.DataFrame(triangle_Dict)
    # result=np.array(result)
    # newres = np.vstack([row[0] for row in result])
    # triangle_Data=np.array(triangle_Data)

    return triangle_Data,result


# use jaccard accuracy
def similarity_calculate(graph,edges):
    # graph=graph.to_undirected()
    result_list = []
    test_list=[]

    for u, v, data in edges:
        nodelist = []
        simlist=[]
        valid_common_neighbors =  set(graph.predecessors(v))
        # valid_common_neighbors = [n for n in neighbors_v if graph.has_edge(n, v)]

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
    with open('test_list.json', 'w') as file:
        json.dump(test_list, file)
    with open('result_list.json', 'w') as file:
        json.dump(result_list, file)
    print(len(test_list))
    print(len(result_list))
    correct = sum(1 for p, t in zip(test_list, result_list) if p == t)
    non_zero_count = len(list(filter(lambda x: x != 0, test_list)))
    print(non_zero_count)
    print(correct)
    accuracy = correct/non_zero_count




    return accuracy









if __name__ == "__main__":
  graph=load_data('soc-sign-epinions/soc-sign-epinions.txt')
  # first_node = list(graph.nodes())[1]
  # feature_list=count_features(graph)
  edges=graph.edges(data=True)
  # print(edges_list[0][2]['label'])

  # 7 features
  # data, result= create_data(graph,edges)
  # data.to_csv('data.csv', index=False)
  # result.to_csv('result.csv', index=False)
  data = pd.read_csv('data.csv')
  # print(data.keys())
  result = pd.read_csv('result.csv')
  # print(result)
  # print(data.shape)
  # print(result.shape)
  # print("7 features")
  # creat_model(data,result)
  # creat_tree_model(data,result)
  # # creat_svm_model(data,result)
  # creat_forest_model(data,result)


  # 16 features
  # data,result=creat_triangle(graph,edges)
  # # data.to_csv('data_triangle.csv', index=False)
  # # result.to_csv('result_triangle.csv', index=False)
  # # print(data[:10])
  # data=data['triangle_Dict'].apply(pd.Series)

  # print("triangle features")
  # creat_model(data, result)
  # # creat_svm_model(data,result)
  # creat_tree_model(data,result)
  # creat_forest_model(data,result)

  # jaccard accuracy
  print(similarity_calculate(graph,edges))


  # print(feature_list[10],result_list[10])
  # print(type(edges_list))


