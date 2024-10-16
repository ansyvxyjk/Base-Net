import numpy as np
import os
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


#过滤数据
def filt(raw_data_path,filt_length,key_sites_tuple,x_pre_base_tuple,y_pre_base_tuple,save_path):
    #长度过滤

    fa = pd.read_csv(raw_data_path, sep='\t')
    print(f'原始数据的序列一共有【{len(fa)}】条。')
    data = fa[fa['Length']==filt_length]
    print(f'因为【序列长度】被过滤了【{len(fa)-len(data)}】条。')
    #根据关键位点过滤（将带有错误位点的序列移除）
    bo_lst = []
    x_pre_1,x_pre_2,x_su_1,x_su_2,y_pre_1,y_pre_2,y_su_1,y_su_2 = key_sites_tuple
    for i in data['Seq']:
        bool_ = (i[x_pre_1:x_pre_2] in x_pre_base_tuple) and (i[y_pre_1:y_pre_2] in y_pre_base_tuple)
        bo_lst.append(bool_)
    data_filt = data[bo_lst]
    print(f'因为【错误序列】被过滤了【{len(data) - len(data_filt)}】条。')
    data_filt.to_csv(save_path,sep='\t')

#加载数据
def load_datasets(sequencing_data_file_path):
    sequencing_data_array = pd.read_csv(sequencing_data_file_path,sep='\t')
    return sequencing_data_array
#将字典保存到csv
def save_dict_to_csv(dict_name,save_path,key_name,value_name):
    df = pd.DataFrame()
    df[key_name] = list(dict_name.keys())
    df[value_name] = list(dict_name.values())
    df.to_csv(save_path,index=False,sep='\t')
#对序列编码
def encode_sequencing_data_and_save_encoded_file(sequencing_data_file_path,key_sites_tuple,encoded_file_save_path):
    #如果存在了已编码的文件路径，则读取文件，否则生成文件
    xy_concat_df_path = os.path.join(encoded_file_save_path, 'xy_concat_df.pkl')
    if os.path.exists(xy_concat_df_path):
         xy_concat_df = pd.read_pickle(xy_concat_df_path)
         return xy_concat_df
    else:
        sequencing_data_array = load_datasets(sequencing_data_file_path)
        x_prefix_dict = {}
        y_prefix_dict = {}
        x_suffix_dict = defaultdict(dict)
        y_suffix_dict = defaultdict(dict)
        prefix_counter = 1
        xy_concat_lst = []
        x_pre_1, x_pre_2, x_su_1, x_su_2, y_pre_1, y_pre_2, y_su_1, y_su_2 = key_sites_tuple
        for base_data in sequencing_data_array['Seq']:
            x_prefix,x_suffic = base_data[x_pre_1:x_pre_2], base_data[x_su_1:x_su_2]
            y_prefix,y_suffic = base_data[y_pre_1:y_pre_2], base_data[y_su_1:y_su_2]


            if x_prefix not in x_prefix_dict:
                x_prefix_dict[x_prefix] = chr(64+prefix_counter)
                prefix_counter+=1

            if y_prefix not in y_prefix_dict:
                y_prefix_dict[y_prefix] = chr(64+prefix_counter)
                prefix_counter+=1

            if x_suffic not in x_suffix_dict[x_prefix_dict[x_prefix]]:
                x_suffix_dict[x_prefix_dict[x_prefix]][x_suffic] = x_prefix_dict[x_prefix] + str(len(x_suffix_dict[x_prefix_dict[x_prefix]]) + 1)
            if y_suffic not in y_suffix_dict[y_prefix_dict[y_prefix]]:
                y_suffix_dict[y_prefix_dict[y_prefix]][y_suffic] = y_prefix_dict[y_prefix] + str(len(y_suffix_dict[y_prefix_dict[y_prefix]]) + 1)
            xy_concat_lst.append((x_suffix_dict[x_prefix_dict[x_prefix]][x_suffic], y_suffix_dict[y_prefix_dict[y_prefix]][y_suffic]))
        save_dict_to_csv(x_prefix_dict, encoded_file_save_path + '/' + 'x_prefix_encoded.xlsx',
                         key_name='x_prefix_sequence', value_name='x_prefix_encoded_id')
        save_dict_to_csv(y_prefix_dict, encoded_file_save_path + '/' + 'y_prefix_encoded.xlsx',
                         key_name='y_prefix_sequence', value_name='y_prefix_encoded_id')
        for i in x_suffix_dict:
            save_dict_to_csv(x_suffix_dict[i], encoded_file_save_path + '/' + f'x_suffix_encoded({i}).xlsx',
                             key_name=f'x_suffix_sequence({i})', value_name=f'x_suffix_encoded_id({i})')
        for i in y_suffix_dict:
            save_dict_to_csv(y_suffix_dict[i], encoded_file_save_path + '/' + f'y_suffix_encoded({i}).xlsx',
                             key_name=f'y_suffix_sequence({i})', value_name=f'y_suffix_encoded_id({i})')
        xy_concat_df = pd.DataFrame(xy_concat_lst,columns=['x_id','y_id'])
        xy_concat_df.to_pickle(xy_concat_df_path)
        return xy_concat_df

#由连接关系xy_concat_df构建图网络
def structure_graph(xy_concat_df):
    G = nx.Graph()
    edges = xy_concat_df.values
    for u,v in edges:
        G.add_edge(u, v)

    return G


#分组（每个细胞上所携带的id信息）的四个方案
#第一个方案
def add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)
def dfs(node, visited, graph):
    cluster = []
    stack = [node]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            cluster.append(vertex)
            stack.extend(graph[vertex])
    return cluster
def find_clusters(edges):
    graph = defaultdict(list)
    for u,v in edges:
        add_edge(graph, u, v)
    visited = set()
    clusters = []
    for node in graph:
        if node not in visited:
            clusters.append(dfs(node, visited, graph))
    return clusters

#第二个方案递归融合求解
def fusion_concat_sets(concat_sets_lst):
    fusioned_sets_lst = []
    for unfusioned_set_item in concat_sets_lst:
        for fusioned_set_item in fusioned_sets_lst:
            if fusioned_set_item&unfusioned_set_item:
                fusioned_set_item.update(unfusioned_set_item)
                break
        else:
            fusioned_sets_lst.append(unfusioned_set_item)
    return fusioned_sets_lst
def recursive_fusion(concat_sets_lst,N):
    if len(concat_sets_lst)<N:
        return fusion_concat_sets(concat_sets_lst)
    grouped_set_lst = [concat_sets_lst[i:i + N] for i in range(0, len(concat_sets_lst), N)]
    fusioned_concat_sets_lst=[]
    for item in grouped_set_lst:
        x = fusion_concat_sets(item)
        fusioned_concat_sets_lst+=x
    return recursive_fusion(fusioned_concat_sets_lst, N)
#第三个方案：先使用pd根据x的信息融合生成中间连接关系列表再由融合算法进行融合
def fusion_df(xy_concat_df:pd.DataFrame):
    x_id = np.unique(xy_concat_df['x_id'].values)
    xy_id=[]
    for i in x_id:
        y_id = set(np.append(xy_concat_df['y_id'][xy_concat_df['x_id']==i].values,i))
        xy_id.append(y_id)
    clusters = fusion_concat_sets(xy_id)
    return clusters
#第四个方案：由图获取clusters

def construct_subgraph(G:nx.Graph,):
    clusters=[]
    set1 = set()
    for node in G.nodes:
        for i in clusters:
            if i.issubset(set1) is False:
                set1.update(i)
                break
        if node not in set1:
            clusters.append(nx.node_connected_component(G,node))
    return clusters
def group_into_cluster(xy_concat_df:pd.DataFrame,G,solution=1,N=10000):
    if solution == 1:
        edges = xy_concat_df.values
        clusters = find_clusters(edges)
    elif solution==2:
        lst1=[]
        for i in xy_concat_df.values:
            lst1.append(set(i))
        clusters = recursive_fusion(lst1, N)
    elif solution==3:
        clusters = fusion_df(xy_concat_df)
    else:
        clusters = construct_subgraph(G)
    return clusters




# 作图
def add_color_to_each_cluster(G:nx.Graph,clusters,cluster_color_lst):
    cluster_color_lst=cluster_color_lst
    node_color_lst=[]
    for node in G.nodes():
        for step,cluster in enumerate(clusters):
            if node in cluster:
                if step < len(cluster_color_lst):
                    node_color_lst.append(cluster_color_lst[step])
                else:
                    node_color_lst.append('gray')
                break
    return node_color_lst

def get_subgraph(G,node_lst):
    return G.subgraph(node_lst)

# def plt_colored_graph_1(G,clusters,graph_save_path,plt_cluster_num=6,with_labels=True,node_size=10,width=0.6,edge_color='gray'):
#     sorted_clusters = sorted(clusters, key=lambda i: len(i), reverse=True)
#     plt_cluster = sorted_clusters[:plt_cluster_num]
#     merge_cluster = sum(plt_cluster,[])
#     g = get_subgraph(G,merge_cluster)
#     node_color_lst = add_color_to_each_cluster(g,plt_cluster,cluster_color_lst=params.cluster_color_lst)
#     pos = nx.spring_layout(g,k=0.02)
#     nx.draw(g,
#             pos,
#             node_size=300,
#             node_color=node_color_lst,
#             with_labels=with_labels,
#             font_size=5,
#             width=1,
#             edge_color=edge_color,
#             alpha=0.6)
#     plt.title("graph-g")
#     plt.savefig(graph_save_path)
#     plt.show()
def plt_colored_graph(G,clusters,plt_config):
    sorted_clusters = sorted(clusters, key=lambda i: len(i), reverse=True)
    plt_cluster = sorted_clusters[:plt_config['plt_cluster_num']]
    merge_cluster = sum(plt_cluster,[])
    g = get_subgraph(G,merge_cluster)

    node_color_lst = add_color_to_each_cluster(g,plt_cluster,cluster_color_lst=plt_config['cluster_color_lst'])
    pos = nx.spring_layout(g,k=plt_config['k'],iterations=plt_config['iterations'])
    nx.draw(g,
            pos,
            node_size=plt_config['node_size'],
            node_color=node_color_lst,
            with_labels=plt_config['with_labels'],
            font_size=plt_config['font_size'],
            width=plt_config['width'],
            edge_color=plt_config['edge_color'])
    plt.title(plt_config['title'])
    plt.savefig(plt_config['graph_save_path'])
    plt.show()
    plt.close()


#主函数

def main():
    #设置全局变量
    global key_sites_tuple,x_pre_base_tuple,y_pre_base_tuple, file_length
    #1设置参数
    #1.1选择第几次数据 例如：First Second ...
    data_name = "Second"
    #1.2根据选择的数据不同，关键位点与碱基序列也不同
    if data_name == "First":
        key_sites_tuple = (0,4,4,24,39,43,43,63)
        x_pre_base_tuple = ('ATTA','ACCG')
        y_pre_base_tuple = ('CATT','CAAG')
        file_length = 63
    elif data_name == "Second":
        key_sites_tuple = (39,44,15,39,59,64,64,88)
        x_pre_base_tuple = ('CAAGC', 'GCTAC')
        y_pre_base_tuple = ('TAATG', 'CTAAG')
        file_length = 103

    #1.3原始数据所在的路径
    raw_data_path = f"../data/raw_data/{data_name}_raw_data.txt"
    #1.4过滤后的数据保存路径
    filt_data_path = f"../data/pro_data/{data_name}_filt_data.txt"
    #1.5编码文件保存的文件夹
    encode_data_save_dir = f"../data/encoded_data/{data_name}"
    #1.6图网络可视化保存路径
    detailed_graph_save_path = f"../data/encoded_data/{data_name}/detailed_graph.jpg"
    rough_graph_save_path = f"../data/encoded_data/{data_name}/rough_graph.jpg"
    #1.7颜色列表
    cluster_color_lst = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    #1.9画图配置
    detailed_plt_config = {'title': 'Detailed_Graph',
                           'graph_save_path': detailed_graph_save_path,
                           'plt_cluster_num': len(cluster_color_lst),
                           'cluster_color_lst': cluster_color_lst,
                           'k': None,
                           'iterations': 100,
                           'node_size': 200,
                           'font_size': 5,
                           'width': 1,
                           'edge_color': 'gray',
                           'with_labels': True,
                           }
    rough_plt_config = { 'title': 'Rough_Graph',
                        'graph_save_path': rough_graph_save_path,
                        'plt_cluster_num': 50,
                        'cluster_color_lst': cluster_color_lst,
                        'k': 0.04,
                        'iterations': 400,
                        'node_size': 10,
                        'font_size': 0.1,
                        'width': 1,
                        'edge_color': 'gray',
                        'with_labels': False,
                         }
    # 1.8簇节点保存路径
    clusters_node_save_path = f"../data/encoded_data/{data_name}/clusters.txt"

    # 开始记录时间
    start_time = time.time()
    #2.1过滤数据
    filt(raw_data_path,file_length,key_sites_tuple,x_pre_base_tuple,y_pre_base_tuple,filt_data_path)
    #2.2根据过滤后的数据 对序列编码 并生成短的连接集合
    xy_concat_df = encode_sequencing_data_and_save_encoded_file(filt_data_path,key_sites_tuple,encode_data_save_dir)
    encode_concat_time = time.time()
    print('过滤数据和序列编码的时间：%.3f秒'%(encode_concat_time-start_time))

    #3根据短的连接关系使用Networkx库可以很简单的将短的连接集合生成簇
    G = structure_graph(xy_concat_df)
    connect_xy_concat_time = time.time()
    print('构建图网络的时间：%.3f秒'%(connect_xy_concat_time-encode_concat_time))

    #4获取每个簇包含的节点（为给簇上颜色做准备）
    #4.1首先要获取每个簇包含的节点，设计了四种方法，现在使用第一种方法
    clusters = group_into_cluster(xy_concat_df, G, solution=1)
    #4.2统计不同节点数量的簇的数量以及频率（
    len_lst = []
    for i in clusters:
        len_lst.append(len(i))
    len_set = set(len_lst)
    len_dict ={}
    for item in len_set:
        len_dict.update({item:len_lst.count(item)})
    print(len_dict)
    sum_num = sum(list(len_dict.values()))
    result = {key:value/sum_num for key,value in len_dict.items()}
    print(result)
    group_into_cluster_time = time.time()
    print('获取每个簇包含的节点：%.5f秒'%(group_into_cluster_time-connect_xy_concat_time))

    '''
    5将图网络可视化（因为簇太多颜色太少的缘故（6种颜色，可以更多但仍不能为所有簇分配不同的颜色），
    因此第一次作图只做前6个最复杂的簇（节点最多的簇），
    第二次作图会尽力画出更多的簇（前6个簇依然是彩色的、其余的是灰色的）
    '''
    plt_colored_graph(G,clusters,detailed_plt_config)
    plt_colored_graph(G,clusters,rough_plt_config)

    #6打印前10个最复杂簇的节点，并保存簇节点文件
    clusters_sorted = sorted_clusters = sorted(clusters, key=lambda i: len(i), reverse=True)
    clusters_df = pd.DataFrame(clusters_sorted)
    print(clusters_df[:10])
    clusters_df.to_csv(clusters_node_save_path,sep='\t')

if __name__ =='__main__':
    main()











