
import copy
from Enhanced_BLS import *
import numpy as np
import time

class BELS():
    def __init__(self, flag, initial_chunk,my_range, feature_size, label_size, chunk_size, N1,N2, N3,preprocess):
        self.flag = flag
        self.my_data_len = initial_chunk
        self.my_range = my_range
        self.feature_size = feature_size
        self.label_size = label_size
        self.preprocess = preprocess
        self.data_numbers = chunk_size
        self.max_learners = 75
        self.N1 = N1
        
        self.N2 = N2
        self.N3 = N3 # of enhancement nodes -----Enhance layer
        
        
        self.s = 0.8  #  shrink coefficient
        self.C = 2**-30 # Regularization coefficient
        self.counter = 0
        
        self.batch_acc_final_results = []
        
        self.x_list = []
        self.y_list = []
        
        self.learners = []
        self.BLS = Enhanced_BLS()
        self.acc_BLS = Enhanced_BLS()
        
        self.y_list_not_encoded = []
        
        self.my_accs = []

        self.initialized = False
        self.beta_1 = []
        self.learners_count = 0
        
        self.data_counter = 0
        
        self.model_test_train_time_start = time.time()
        self.preds = np.zeros((self.data_numbers,self.label_size))
        self.batch_acc_total = []
        
        self.ensemble_accs = []
        
        self.current_learner = 0
        
        self.weight_data = np.zeros((1 , (((self.N1*self.N2) + self.N3)*3) + 2))
        
        self.there_is_full_accurate = False
        self.worst_list = []
        self.good_list = []
        
        self.worst_list_counter = []
        self.restarted_list = []
        
        self.pred_prob = np.zeros((self.data_numbers,self.label_size))
        self.eleminated = []
        self.learning_second_phase_time = 0 
        self.old_besties = []
        self.threshold = 50
        self.batch_kappa = []
    
    def test_then_train(self, data, labels, encodings):
        if(self.flag == True):
            self.my_data_len = len(data)
        for i in range(0,self.my_data_len):
        
            
            mydata = []
            X = data[i]
            for j in range(0,self.feature_size ):
                mydata.append(float(X[j]))
            mydata = np.array(mydata)
        
            mydata = mydata.flatten()
            mydata = mydata.astype(float)
            y = labels[i]
            y = int(y)
            y1 = y
            y1 = str(y1) 
            y =str(y)
            y = encodings[y]
        
            self.counter += 1
            
            if(len(self.x_list) < self.data_numbers - 1 and i +1 < len(data)):
                
                self.x_list.append(mydata)
                self.y_list.append(y)
                self.y_list_not_encoded.append(y1)
        
            elif(len(self.x_list) >= (self.data_numbers - 1) or i +1 == len(data)):
                
                self.x_list.append(mydata)
                
                
                self.y_list.append(y)
                self.y_list_not_encoded.append(y1)
               

        
                self.x_list = np.array(self.x_list)
                self.y_list = np.array(self.y_list)
                self.y_list_not_encoded = np.array(self.y_list_not_encoded)
        
            
                self.x_list = self.x_list.reshape(self.x_list.shape[0],-1)
                self.y_list = self.y_list.reshape(self.y_list.shape[0],-1)
                
                self.y_list_not_encoded = self.y_list_not_encoded.reshape(self.y_list.shape[0],-1)
         
                self.preds = np.zeros((len(self.x_list),self.label_size))
                self.pred_prob = np.zeros((len(self.x_list),self.label_size))
                
                if(self.learners_count < self.max_learners):  
                    
                    Enhanced_BLS_tmp = None
                    Enhanced_BLS_tmp = Enhanced_BLS() 
                    #print("adding new learner")
                    self.restarted_list = []
                    self.learners.append(Enhanced_BLS_tmp)
                    self.current_learner = self.learners_count# + 2
                    self.learners_count += 1 
                    self.restarted_list.append(self.current_learner)
                    
                    #print("number of self.learners: ", self.learners_count)
                if(self.learners_count >= self.max_learners or  (len(self.worst_list) > len(self.learners)/2) ):
                    
                    self.restarted_list = []
                    if(len(self.worst_list) != 0 or ((len(self.bestes_list) == 0 and len(self.good_list) == 0))):
                        
                        for k in range(0 ,int(((len(self.worst_list))) -1 )):
                            
                            
                            l = self.worst_list[k]
                            if(len(self.worst_list) > len(self.learners)/2 and l!=0):
                                
                                self.eleminated.append(self.learners[l])
                            
                            if(l!=0  ):
                                                            
                                self.learners[l] = 0
                       
                        learners_tmp_new = []    
                       
                        for k in range(0 , len(self.learners)):
                            if(self.learners[k] != 0):
                                
                                learners_tmp_new.append(self.learners[k])
                        self.learners = learners_tmp_new
                        
                        ff = 0
                        
                        while (len(self.learners) < (self.max_learners ) and ff < len(self.old_besties)):
                            if(self.old_besties[ff] < len(self.eleminated)):
                                tmpppp = copy.deepcopy(self.eleminated[self.old_besties[ff]])
                           
                                self.learners.append(tmpppp)
                                
                                del self.eleminated[self.old_besties[ff]]
            
                                self.restarted_list = []
                                
                            ff += 1
        
                        self.old_besties = []
                        
                        if(len(self.eleminated ) >300):
                            self.eleminated = self.eleminated[20:300]
                        
                        self.learners_count = len(self.learners)
                
                
                self.worst_list = []
                self.bestes_list = []
                self.good_list = []
                self.data_counter = self.data_counter + len(self.x_list)
                
                if(self.initialized == False ):
        
                    self.beta_1 = self.BLS.BLS_online_init(self.preprocess, self.x_list,self.s,self.C,self.N1,self.N2,self.N3)           
                    for m in range(0 , self.learners_count):      
                                        
                        self.learners[m].update_second_layer(self.beta_1,self.y_list,self.C)
                    self.initialized = True
        
                elif(self.initialized == True ):
                   
                    TT3 = self.BLS.test_first_phase(self.preprocess, self.x_list)         
                
                    testing_count = 0
                    for m in range(0 , self.learners_count):
                        
                        if((m not in self.restarted_list)):
                            testing_count += 1
                        
                            if(len(self.learners[m].beta2 ) > 0):
                                self.my_accs ,batc_acc ,m_pred = self.learners[m].test_second_phase(TT3,self.y_list)
                                
                                if(self.data_numbers != 2 and len(self.ensemble_accs)!=0):                            
                                
                                    self.threshold = self.ensemble_accs[len(self.ensemble_accs)-1]
                                
                                
                                
                                if(batc_acc == 100):
                                    self.there_is_full_accurate = True
                                    
                                    self.bestes_list.append(m)
                                    
                                if( batc_acc < 100 and batc_acc > self.threshold):
                                    self.there_is_full_accurate = True
                                    
                                    self.good_list.append(m)
                             
                                if(batc_acc < self.threshold) :
                                    self.there_is_full_accurate = True
                                    self.worst_list.append(m)
                   
                                prediction_prob = m_pred
                                
                                prediction = m_pred
                            
                                prediction_mins = np.zeros_like(prediction)
                      
                                prediction_mins[np.arange(len(prediction)), prediction.argmax(1)] = 1
                    
                                prediction = 1 * prediction_mins 
                                
                                
                                self.preds = self.preds + prediction
                                
                                self.pred_prob = self.pred_prob +prediction_prob
         
                    for jj in range(0 , len(self.preds)):
                        if(self.preds[jj,0] == self.preds[jj,1]):
                            self.preds[jj] = self.pred_prob[jj]
            
                    self.old_besties = []
                    
                    for ll in range(0 , len(self.eleminated)):
                        
                        my_accs_e ,batc_acc_e ,m_pred_e = self.eleminated[ll].test_second_phase(TT3,self.y_list)
                        if(batc_acc_e > 50):
                            self.old_besties.append(ll)
                    
                    
                    if(self.there_is_full_accurate == True):
                        self.there_is_full_accurate = False
           
                    self.worst_list_counter.append(len(self.worst_list))
          
                    wrong_index = []
                    _, ensemble_acc , tmp_batch_acc , wrong_index = self.acc_BLS.show_accuracy_ensemble(self.preds,self.y_list,len(self.x_list))
                    
                    check_kappa= False
                    if(i + self.data_numbers > (self.my_data_len)):
                        check_kappa = True
                    kappa = self.acc_BLS.kappa_s(self.preds,self.y_list,check_kappa)
                    if(kappa != -1):
                        self.batch_kappa.append(kappa)
                    
                    self.batch_acc_final_results.append(tmp_batch_acc)
                   
                    self.batch_acc_total.append(tmp_batch_acc)
                    ensemble_acc = ensemble_acc * 100
                    self.ensemble_accs.append(ensemble_acc)
                    if(self.flag):
                        print('-------------------Online_BLS---------------------------')

                        print("Kappa: ",kappa)
                        print("ensemble acc: " , ensemble_acc)

                        print(i)
                  
                    self.beta_1 = self.BLS.update_first_layer(self.preprocess, self.x_list)
                    
                    for zz in range(0 , self.learners_count):
                        
                        second_phase_start = time.time()
                        self.learners[zz].update_second_layer(self.beta_1,self.y_list,self.C)
                        self.learning_second_phase_time += (time.time() - second_phase_start) 
                        
                self.x_list = []
                self.y_list = []
                self.y_list_not_encoded = []
        
                
        model_test_train_time_end = time.time()
        
        total_time = model_test_train_time_end - self.model_test_train_time_start 
        average_kappa = np.mean(self.batch_kappa[1:])
    
        self.batch_kappa = self.batch_kappa[1:]
        
        self.batch_acc_final_results = self.batch_acc_final_results[self.my_range-1:]
        
        return self.batch_kappa,average_kappa,total_time, np.mean(self.batch_acc_final_results), self.batch_acc_final_results
