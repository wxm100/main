import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only WARNING and ERROR

from tensorflow.lite.python.util import is_frozen_graph


# def load_data(path):
#    G=nx.DiGraph()
#    with open(path, 'r') as f:
#       for _ in range(4):
#          next(f)
#       for line in f:
#          from_node,to_node,attitude=line.split()
#          G.add_edge(from_node,to_node,label=attitude)

#       return G

def load_data(path):

    data = pd.read_csv(path, header=None, names=['from_node', 'to_node', 'label'])
    
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row['from_node'], row['to_node'], label=row['label'])
    return G


def count_features(G):
   # Eigenvector Centrality
   # eigenvector_centrality = nx.eigenvector_centrality(G)
   eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)
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

        # Eigenvector Centrality
       centrality = eigenvector_centrality[node]
       
       Node_feature_list.append({str(node):{
                                            'D_in':D_in,
                                            'D_out':D_out,
                                            'In_label_pos':In_label_pos,
                                            'In_label_neg':In_label_neg,
                                            'Out_label_pos':Out_label_pos,
                                            'Out_label_neg':Out_label_neg,
                                            'Eigenvector_Centrality': centrality
                                            }})
   return Node_feature_list

def calculate_embeddedness(graph,u,v):
   if not graph.has_edge(u, v):
      return 0

   neighbors_u = set(graph.neighbors(u))
   neighbors_v = set(graph.neighbors(v))
   common_neighbors = neighbors_u & neighbors_v

   return len(common_neighbors)


def create_data(graph,edges):
    feature_list=[]   ######
    result_list= []
    D_in_list=[]
    D_out_list=[]
    In_label_pos_list=[]
    In_label_neg_list=[]
    Out_label_pos_list=[]
    Out_label_neg_list=[]
    embed_list=[]

    eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)

    centrality_u_list = []
    centrality_v_list = []


    for u,v,data in edges:
        embed=calculate_embeddedness(graph,u,v)

        centrality_u = eigenvector_centrality[u]
        centrality_v = eigenvector_centrality[v]

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

        D_in_list.append(D_in)
        D_out_list.append(D_out)
        In_label_pos_list.append(In_label_pos)
        In_label_neg_list.append(In_label_neg)
        Out_label_pos_list.append(Out_label_pos)
        Out_label_neg_list.append(Out_label_neg)
        embed_list.append(embed)

        centrality_u_list.append(centrality_u)
        centrality_v_list.append(centrality_v)

        result_list.append(data['label'])
        
    feature_Dict={
                  'D_in':D_in_list,
                  'D_out':D_out_list,
                  'In_label_pos':In_label_pos_list,
                  'In_label_neg':In_label_neg_list,
                  'Out_label_pos':Out_label_pos_list,
                  'Out_label_neg':Out_label_neg_list,
                  'embed_list':embed_list,
                  'centrality_u': centrality_u_list,
                  'centrality_v': centrality_v_list,
                  }
    result_Dict={'result':result_list}
    data = pd.DataFrame(feature_Dict)
    result=pd.DataFrame(result_Dict)

    print(result.shape)

    return data,result


def creat_model(data,result):

    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)

    # shape to 1
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    model = LogisticRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)


if __name__ == "__main__":
  
  graph=load_data('simplified_soc-sign-epinions.csv')

  edges=graph.edges(data=True)

  data, result= create_data(graph,edges)
  data.to_csv('data.csv', index=False)
  result.to_csv('result.csv', index=False)

  creat_model(data,result)