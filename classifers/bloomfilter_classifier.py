import pandas as pd
from pandas import DataFrame
from utality import clustering
from pybloom import BloomFilter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

TRAINING = 101
TEST = 102
VALIDATION = 103
ABANDON = 104

'the dataset is from http://www.ece.uah.edu/~thm0009/icsdatasets/IanArffDataset.arff'
'since the dataset is flawed with attacks are mislabeled for commands and responses, we combine a command and a response to a single package for classification'
data = pd.read_csv("../data/pruned_gas_data.csv")




'sampling to make sure the training set capture the normal operational profile of gas pipeline'
sampling = random.sample(range(0,data.shape[0]), data.shape[0])
training_samples = sampling[:len(sampling)*3//5]
validation_samples = sampling[len(sampling)*3//5:len(sampling)*4//5]
testing_samples = sampling[len(sampling)*4//5:]

data['tag'] = TRAINING
data.loc[validation_samples, 'tag'] = VALIDATION
data.loc[testing_samples, 'tag'] = TEST
data.loc[(data['tag'] != TEST) & (data['binary result']==1), 'tag'] = ABANDON

'remove anomalies from training and validation dataset'
training_data = data[data['tag'] == TRAINING]
training_data = training_data[training_data['binary result'] == 0] 

validation_data = data[data['tag'] == VALIDATION]
validation_data = validation_data[validation_data['binary result'] == 0]
validation_samples = validation_data.index.values.tolist()

test_data = data[data['tag'] == TEST]

 
def package_level_dectection(training_data, data):
   
    clustering._clusterDataColumnNoQ(training_data, data,'interval_1',2)
    clustering._clusterDataColumnNoQ(training_data, data,'interval_2',2)
    
    clustering._divideDataColumnWithQ(training_data, data,'setpoint',9,standard=0)
    clustering._divideDataColumnWithQ(training_data, data,'pressure measurement',19,standard=0)
    
    'standard=1 --> only is used for detecting outliers'
    clustering._divideDataColumnWithQ(training_data, data,'gain',2,standard=1)
    clustering._divideDataColumnWithQ(training_data, data,'reset rate',2,standard=1)
    clustering._divideDataColumnWithQ(training_data, data,'rate',1,standard=1)
    clustering._divideDataColumnWithQ(training_data, data,'deadband',2,standard=1)
    clustering._divideDataColumnWithQ(training_data, data,'cycle time',2,standard=1)
    
    clustering._clusterMulDimensionalDataColumn(training_data, data, ['gain','reset rate', 'deadband', 'cycle time', 'rate'], 'PID_cluster', 31)
    clustering._clusterDataColumnNoQ(training_data, data,'cmd_crc rate',2)
    clustering._clusterDataColumnNoQ(training_data, data,'rps_crc rate',2)
  
    
    ##### clear unused features
    specific_result = data['specific result']
    binary_result = data['binary result']
    tag = data['tag']
    
    for entry in ['interval_1','interval_2','setpoint', 'reset rate', 'gain', 'rate',
                  'deadband','cycle time','pressure measurement','cmd_crc rate','rps_crc rate',
                  'binary result', 'specific result', 'tag']:
        training_data = training_data.drop(entry, 1)
        data = data.drop(entry, 1)
    data = data.drop('valid',1)    
    
    ##################### construct bloom filter
    f = BloomFilter(capacity=training_data.shape[0], error_rate=0.0001)
          
    x = training_data.to_string(header=False,
                      index=False,
                      index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in x]
      
    for i in range(0,training_data.shape[0]):
        f.add(vals[i])
    
    ################### check validation error          
    x = data.to_string(header=False,
                         index=False,
                         index_names=False).split('\n')
    data_vals = [','.join(ele.split()) for ele in x]
     
    filter_result = []
    for i in range(data.shape[0]):
        if(data_vals[i] in f):
            filter_result.append(0)
        else:
            filter_result.append(1)
                
    validation_result = [filter_result[i] for i in validation_samples]
    
    validation_err = validation_result.count(1)*1.0/len(validation_result)
    
    print( 'validation error:' + str(validation_err) )
    

    ####construct signature database 
    for entry in ['reset rate', 'gain', 'rate','deadband','cycle time']:
        data = data.drop(entry + '_cluster', 1)
        
    x = data.to_string(header=False,
                         index=False,
                         index_names=False).split('\n')
    data_vals = [','.join(ele.split()) for ele in x]
     
     
    num=0
    dict = {} 
    for i in range(len(data_vals)):
        if(filter_result[i] == 1):
            data_vals[i] = '?'
        else:
            if data_vals[i] in dict:
                data_vals[i] = dict[data_vals[i]]
            else:
                dict[data_vals[i]] = num
                data_vals[i] = num
                num += 1
                               
    df = {'signature': data_vals, 'binary result': binary_result, 'specific result': specific_result, 'bf result':filter_result, 'tag':tag }
    df = DataFrame(data=df)
    df = pd.concat([df, data], axis=1)
    df.to_csv("../data/filtered_data.csv", index=False)
      

package_level_dectection(training_data, data)
