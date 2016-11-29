# Seizure Prediction
Implement simple-rnn/residual-rnn for kaggle competitions:
</br>
</br>
Melbourne University AES/MathWorks/NIH Seizure Prediction: 
</br> 
<https://www.kaggle.com/c/melbourne-university-seizure-prediction>

---
##0. Preprocess
Generate file list
```C
$ ls train_1/ *_0.mat >> train1_neg.list
$ ls train_1/ *_1.mat >> train1_pos.list
```
</br>
Use following script to check damage file.
```C
$ python quick_data_check_train.py
$ python quick_data_check_test.py
```
</br>
##1. Simple RNN by python script
**Folder:** Simple_RNN
</br>
**Train:**
```
$ python simple_rnn_1.py # for set-1
$ python simple_rnn_2.py # for set-2
$ python simple_rnn_3.py # for set-3
```
**Test:**
```
$ python predict_simple_rnn.py
```
**Model:**
</br>
There are some trained model under model folder
</br>
</br>
</br>
</br>
##2. Residual RNN by python script
**Folder:** Res_RNN
</br>
**Train:**
```
$ python res_rnn_1.py # for set-1
$ python res_rnn_2.py # for set-2
$ python res_rnn_3.py # for set-3
```
**Test:**
```
$ python predict_res_rnn.py
```
**Model:**
</br>
There are some trained model under model folder
</br>
</br>
</br>
</br>
**Feel free to download and make it your use-case : )**
