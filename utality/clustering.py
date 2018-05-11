'''
Created on 3 Nov 2016

@author: cf1510
'''
from sklearn.cluster import KMeans

sigma = 0.05

orphan_theta = 500  


def _detect_kmeans_outlier(cluster_centers, max_dist, data, cluster_num, entry_name, new_entry_name):
    data.loc[:,'valid'] = 0
    for i in range(cluster_num):
        data.loc[abs(data[entry_name]-cluster_centers[i][0])<=(1+sigma)*max_dist[i], 'valid'] = 1
    data.loc[data['valid']==0, new_entry_name] = cluster_num
    data = data.drop('valid', 1)

def _clusterDataColumnWithQ(training_data, test_data, entry_name, cluster_num):
    new_entry_name = entry_name + "_cluster"
    
    training_data.loc[training_data[entry_name] == '?', entry_name] = 999
    training_data[entry_name] = training_data[entry_name].astype(float)
    
#     validation_data.loc[validation_data[entry_name] == '?', entry_name] = 999
#     validation_data[entry_name] = validation_data[entry_name].astype(float)
    
    test_data.loc[test_data[entry_name] == '?', entry_name] = 999
    test_data[entry_name] = test_data[entry_name].astype(float)
    
    entry_data = training_data[entry_name].reshape(-1, 1)
    
    kmeans_ret = KMeans(n_clusters=cluster_num).fit(entry_data)
    training_data.loc[:,new_entry_name] = kmeans_ret.labels_
    
    max_dist = [0] * cluster_num
    for i in range(cluster_num):
        max_dist[i] = max(abs(training_data.loc[training_data[new_entry_name] == i , entry_name]-kmeans_ret.cluster_centers_[i][0]))
    
#     validation_data.loc[:,new_entry_name] = kmeans_ret.predict(validation_data[entry_name].reshape(-1,1))
    test_data.loc[:,new_entry_name] = kmeans_ret.predict(test_data[entry_name].reshape(-1,1))
    
    min_data = min(entry_data)
    max_data = max(entry_data)
#     validation_data.loc[(validation_data[entry_name]-max_data*(1+sigma)>0) | (validation_data[entry_name]-min_data*(1-sigma)<0),new_entry_name] = cluster_num
    test_data.loc[(test_data[entry_name]-max_data*(1+sigma)>0) | (test_data[entry_name]-min_data*(1-sigma)<0),new_entry_name] = cluster_num

#     _detect_kmeans_outlier(kmeans_ret.cluster_centers_, max_dist, validation_data, cluster_num, entry_name, new_entry_name)
    _detect_kmeans_outlier(kmeans_ret.cluster_centers_, max_dist, test_data, cluster_num, entry_name, new_entry_name)
    
    
def _detect_even_inteval_outlier(data, new_entry_name, entry_name, max_data, min_data, interval_num, interval):
    data.loc[:,new_entry_name] = 0
    ##outliers     
    data.loc[((data[entry_name]-max_data*(1+sigma)>0) | (data[entry_name]-min_data*(1-sigma)<0)) 
                  & data[entry_name] != 999,new_entry_name] = interval_num+1
    data.loc[data[entry_name] == 999,new_entry_name] = interval_num
    
    for i in range(interval_num):
        data.loc[(data[entry_name]>=min_data+i*interval) & (data[entry_name]<=min_data+(i+1)*interval),new_entry_name] = i
     
    data.loc[(data[entry_name]<=min_data) & (data[entry_name]>=min_data*(1-sigma)),new_entry_name] = 0
    data.loc[(data[entry_name]>=max_data) & (data[entry_name]<=max_data*(1+sigma)),new_entry_name] = interval_num-1
    

def _divideDataColumnWithQ(training_data, test_data, entry_name, interval_num, standard = 1):
    new_entry_name = entry_name + "_cluster"
    
    
    training_data.loc[training_data[entry_name] == '?', entry_name] = 999
    training_data.loc[:,entry_name] = training_data[entry_name].astype(float)
    
#     validation_data.loc[validation_data[entry_name] == '?', entry_name] = 999
#     validation_data.loc[:,entry_name] = validation_data[entry_name].astype(float)
    
    test_data.loc[test_data[entry_name] == '?', entry_name] = 999
    test_data.loc[:,entry_name] = test_data[entry_name].astype(float)
    
    max_data = training_data[entry_name][training_data[entry_name] != 999].max()
    min_data = training_data[entry_name][training_data[entry_name] != 999].min()
    
    interval = float(max_data-min_data)/interval_num
    training_data[new_entry_name] = 0
    training_data.loc[training_data[entry_name] == 999,new_entry_name] = interval_num
    orphans = []
    for i in range(interval_num):
        training_data.loc[(training_data[entry_name]>=min_data+i*interval) & (training_data[entry_name]<=min_data+(i+1)*interval),new_entry_name] = i
        if training_data[training_data[new_entry_name]==i].shape[0] < orphan_theta:
            orphans.append(i)
    
#     _detect_even_inteval_outlier(validation_data, new_entry_name, entry_name, max_data, min_data, interval_num, interval)
    _detect_even_inteval_outlier(test_data, new_entry_name, entry_name, max_data, min_data, interval_num, interval)
      
    if standard == 1:
        training_data.loc[training_data[new_entry_name] < interval_num, new_entry_name] = 0
#         validation_data.loc[validation_data[new_entry_name] < interval_num, new_entry_name] = 0
        test_data.loc[test_data[new_entry_name] < interval_num, new_entry_name] = 0
    else:
        for i in orphans:
            training_data.loc[training_data[new_entry_name]==i,new_entry_name] = orphans[0]
            test_data.loc[test_data[new_entry_name]==i,new_entry_name] = orphans[0]
    

def _clusterMulDimensionalDataColumn(training_data, test_data, entry_list, new_entry_name, cluster_num):
    
    for item in entry_list:
        max_value = training_data[item].max()
        min_value = training_data[item].min()
        training_data.loc[:,item]=training_data[item].apply(lambda x: (x-min_value)/(max_value-min_value))
#         validation_data.loc[:,item]=validation_data[item].apply(lambda x: (x-min_value)/(max_value-min_value))
        test_data.loc[:,item]=test_data[item].apply(lambda x: (x-min_value)/(max_value-min_value))
        
    
    entry_data = training_data[entry_list].as_matrix()
    
    kmeans_ret = KMeans(n_clusters=cluster_num).fit(entry_data)
#     print kmeans_ret.cluster_centers_
    training_data.loc[:,new_entry_name] = kmeans_ret.labels_
     
#     validation_data.loc[:,new_entry_name] = kmeans_ret.predict(validation_data[entry_list].as_matrix())
    
    test_data.loc[:,new_entry_name] = kmeans_ret.predict(test_data[entry_list].as_matrix())
    orphans = []
    for i in range(cluster_num):
        if training_data[training_data[new_entry_name]==i].shape[0] < orphan_theta:
            orphans.append(i)
    for i in orphans:
        training_data.loc[training_data[new_entry_name]==i,new_entry_name] = orphans[0]
        test_data.loc[test_data[new_entry_name]==i,new_entry_name] = orphans[0]

            
    
def _clusterDataColumnNoQ(training_data, test_data, entry_name, cluster_num):
    new_entry_name = entry_name + "_cluster"
     
    entry_data = training_data[entry_name].reshape(-1, 1)
        
    kmeans_ret = KMeans(n_clusters=cluster_num).fit(entry_data)
    
    training_data.loc[:,new_entry_name] = kmeans_ret.labels_
    
    max_dist = [0] * cluster_num
    for i in range(cluster_num):
        max_dist[i] = max(abs(training_data.loc[training_data[new_entry_name] == i , entry_name]-kmeans_ret.cluster_centers_[i][0]))
    
#     validation_data.loc[:,new_entry_name] = kmeans_ret.predict(validation_data[entry_name].reshape(-1,1))
    test_data.loc[:,new_entry_name] = kmeans_ret.predict(test_data[entry_name].reshape(-1,1))
    
    min_data = min(entry_data)
    max_data = max(entry_data)
#     validation_data.loc[(validation_data[entry_name]-max_data*(1+sigma)>0) | (validation_data[entry_name]-min_data*(1-sigma)<0),new_entry_name] = cluster_num
    test_data.loc[(test_data[entry_name]-max_data*(1+sigma)>0) | (test_data[entry_name]-min_data*(1-sigma)<0),new_entry_name] = cluster_num

#     _detect_kmeans_outlier(kmeans_ret.cluster_centers_, max_dist, validation_data, cluster_num, entry_name, new_entry_name)
    _detect_kmeans_outlier(kmeans_ret.cluster_centers_, max_dist, test_data, cluster_num, entry_name, new_entry_name)
    
             
     
    