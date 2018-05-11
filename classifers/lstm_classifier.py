'''
Created on 22 May 2017

@author: cf1510
'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras.metrics as metrics
import random
from keras.callbacks import ModelCheckpoint

TRAINING = 101
TEST = 102
VALIDATION = 103
ABANDON = 104

disc_input = ['cmd_address', 'cmd_function', 'cmd_length', 'cmd_command response',  
              'rps_address', 'rps_function', 'rps_length', 'rps_command response', 
              'control scheme', 'system mode', 'pump', 'solenoid', 'interval_1_cluster', 'interval_2_cluster', 
              'setpoint_cluster','pressure measurement_cluster', 'PID_cluster', 'cmd_crc rate_cluster','rps_crc rate_cluster']

disc_target= ['signature']

lookback = 4
alpha = 10.01

def top_10_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 10)

def top_9_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 9)

def top_8_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 8)

def top_7_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 7)

def top_5_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 5)

def top_6_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 6)

def top_4_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 4)

def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 3)

def top_2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 2)



def _evaluate_prediction(actual_result,predict_result, verbose = 1):
    cmatrix = confusion_matrix(actual_result, predict_result)
    precision = cmatrix[1][1]*1.0/(cmatrix[1][1]+cmatrix[0][1])
    recall = cmatrix[1][1]*1.0/(cmatrix[1][1]+cmatrix[1][0])
    f1score = 2*precision*recall/(precision+recall)    
    accuracy = (cmatrix[1][1]+cmatrix[0][0])*1.0/(cmatrix[1][1]+cmatrix[0][1]+cmatrix[0][0]+cmatrix[1][0])
    if(verbose == 1):
        print( 'precision: ' + str(precision) )
        print( 'recall: ' + str(recall) )
        print( 'f1score: ' + str(f1score) )
        print( 'accuracy: ' + str(accuracy) )
    return precision, recall, f1score, accuracy

def _load_data(dataset, onehot_entries):
    trainX, trainY, validationX, validationY, testX, testY = [], [], [], [], [], []
    binary_result = []
    specific_result = []
    filter_result = []
    
    input_signals = []
    for entry in disc_input:
        input_signals.extend(onehot_entries[entry])
#         inputs.extend(onehot_entries[entry])
#     print input_signals
    
    output_signals = []
    for entry in disc_target:
        output_signals.extend(onehot_entries[entry])
#         inputs.extend(onehot_entries[entry])
#     print output_signals
    
    for i in range(len(dataset)-lookback-2):
        if i % 10000 ==0:
            print(i)
            
        if dataset.loc[i+lookback, 'tag'] == TEST: #test data
            testX.append( dataset.loc[i: i+lookback-1, input_signals].as_matrix() )
            testY.append( dataset.loc[i+lookback, output_signals].as_matrix() )
            binary_result.append(dataset.loc[i+lookback, 'binary result'])
            specific_result.append(dataset.loc[i+lookback, 'specific result'])
            filter_result.append(dataset.loc[i+lookback, 'bf result'])
            
        else:
            dataX_copy = dataset.iloc[i:i+lookback].copy()
            dataX_copy = dataX_copy.reset_index(drop=True)
            binary_rets = list(dataX_copy['binary result'].values)
            if binary_rets.count(1) == 0:##only use normal data as training and validation
                for j in range(lookback):##corruput input
                    if(random.random()<alpha/(alpha+data.loc[i+j,'freq'])):
                        choice = random.randint(0,len(disc_input)-1)
                        chosen_feature = disc_input[choice]
                        feature_name = onehot_entries[chosen_feature]
                            
                        if(len(feature_name) == 1):
                            dataX_copy.loc[j,feature_name] = 1-data.loc[i+j,feature_name]
                        else:
                            feature_vector = [0] * len(feature_name)
                            choice = len(feature_vector)-1
                            feature_vector[choice] = 1                                
                            dataX_copy.loc[j,feature_name] = feature_vector 
            if dataset.loc[i+lookback, 'tag'] == TRAINING:    
                trainX.append( dataX_copy.loc[:, input_signals].as_matrix() )
                trainY.append( dataset.loc[i+lookback, output_signals].as_matrix() )
            if dataset.loc[i+lookback, 'tag'] == VALIDATION:
                validationX.append( dataX_copy.loc[:, input_signals].as_matrix() )
                validationY.append( dataset.loc[i+lookback, output_signals].as_matrix() )  
    
    return np.array(trainX), np.array(trainY), np.array(validationX), np.array(validationY), np.array(testX), np.array(testY), binary_result, specific_result, filter_result


######### 
'read data'
data = pd.read_csv("../data/filtered_data.csv")
print(data.shape)
# data = data[:40000]

data['freq'] = data.groupby('signature')['signature'].transform('count')

onehot_entries = {}

### onehot encoding
'preprocessing'
for entry in disc_input:
    newdf = pd.get_dummies(data[entry]).rename(columns=lambda x: entry + '^' + str(x))
    onehot_entries[entry]= newdf.columns.values.tolist()
    data = pd.concat([data, newdf], axis=1)

for entry in disc_target:
    newdf = pd.get_dummies(data[entry]).rename(columns=lambda x: entry + '^' + str(x))
    onehot_entries[entry]= newdf.columns.values.tolist()
    data = pd.concat([data, newdf], axis=1)

##########
'load data'

trainX, trainY, validationX, validationY, testX, testY, binary_result, specific_result, filter_result = _load_data(data, onehot_entries)     
print(trainX.shape)
print(trainY.shape)
######
'train model'                  
input_dim = trainX.shape[2]
output_dim = trainY.shape[1]
                   
model = Sequential()
model.add(Dropout(0.5,input_shape=(lookback,input_dim))) ##dropout to increase generality
model.add(LSTM(256,  return_sequences=True))  
model.add(LSTM(256, return_sequences=False))
model.add(Dense(output_dim))  
model.add(Activation("softmax"))   
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_2_acc, top_3_acc, top_4_acc, top_5_acc, 
                                                                          top_6_acc, top_7_acc, top_8_acc, top_9_acc,top_10_acc])

model.summary()


##### train the model

checkpointer = ModelCheckpoint(filepath="../data/lstm_classifier.h5", verbose=1, save_best_only=True)
                     

model.fit(trainX, trainY, batch_size=256, nb_epoch=50, verbose=2,validation_data=(validationX,validationY),callbacks=[checkpointer])      

                     
model_json = model.to_json()
with open("../data/lstm_classifier.json", "w") as json_file:
    json_file.write(model_json)

########## load model from config file
'load model from file'
             
json_file = open("../data/lstm_classifier.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../data/lstm_classifier.h5")



########## show test result
y_pred = model.predict_proba( testX, verbose=0 )

actual_result = list(binary_result)
predict_result = list(filter_result)
for K in range(1,11):
    predict_result = list(filter_result)
    for i in range(len(y_pred)):
        if predict_result[i] == 0:
            topK = sorted(range(len(y_pred[i])), key=lambda k: y_pred[i,k], reverse=True)[:K]
#             print y_pred[i,topK]
             
            y_true = testY[i,:]
            
            #     print y_true
            anomaly = True
            for j in topK:
                if y_true[j] == 1:
                    anomaly = False
            
            if anomaly == True:
                predict_result[i] = 1
    
    print()       
    print( 'actual 1: ' + str(actual_result.count(1)) +'  0: '+ str(actual_result.count(0)) )
    print( 'predict 1: ' + str(predict_result.count(1)) +'  0: '+ str(predict_result.count(0)) )
    print( 'k=' + str(K) )       
    precision, recall, f1score, accuracy = _evaluate_prediction(actual_result,predict_result)
    
    type = [[29,30,31,32],[35,27,34,28,26,33,25],[14,16,13,15,17],[12,6,4,11,8,1,3,7,10,2,5,9],[21,19,22],[18],[23,24,20]] 
    hit = [0] * len(type)
    count = [0] * len(type)
    for i in range(len(specific_result)):
        for j in range(len(type)):
            if(specific_result[i] in type[j]):
                count[j] += 1
                if predict_result[i] == 1:
                    hit[j] += 1
    
    for i in range(len(type)):
        print( 'attack type ' + str(i+1) + ' recognize rate: ' + str(hit[i]) + '/' + str(count[i]) )
    
