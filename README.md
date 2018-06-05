## Att-LSTM

- <b>The schematic of the Att-LSTM </b>  
![Model architecture](https://github.com/Tom957/Att-LSTM/blob/master/images/attlstm.jpg)


### Requirements
- Python 3.5.2
- [Tensorflow 0.12][1]

### Datasets
- Adding Problem was proposed by Hochreiter & Schmidhuber, and the dataset is create randomly. 
- A sequential version of the handwritten digit classification (MNIST), which is downloaded by tensorflow.


### Usage

- <b>Training Models</b>
  * Add problem usage ```python test_add.py```
      - <b>Mian Options</b>
      - `batch_size`: Batch size. Default is 20.
      - `step_size`: The length of input, which is called T in my paper. The value is checked in {100,200,400,600}.
      - `input_size`: Dimension of input. Default is 2.
      - `output_size`: Dimension of output. Default is 1.
      - `unit_size`:  The number of hidden unit. Default is 100.
      - `learning_rate`: Learning rate. Default is 0.001.
      - `epoch_num`: Max number of epochs. Default is 600.
      - `cell_name`: Three cell can be choiced, including rnn, lstm, arnn.  Default is arnn.
      - `K`: Hyperparameter for the Att-LSTM. Default is 4.
  * MNIST usage  ``` python test_mnist.py ```and pMNIST usage ```python test_pmnist.py```
      - <b>Mian Options</b>
      - `batch_size`: Batch Size. Default is 100.
      - `step_size`: The length of input. The value number of pixel. Default is 784.
      - `input_size`: Dimension of input. Default is 1.
      - `class_num`: Dimension of output. Default is 10.
      - `unit_size`:  The number of hidden unit. Default is 1000.
      - `learning_rate`: Learning Rate. Default is 0.001.
      - `clip_value`: Clip gradient and avoid gradient explosion. Default is 1.0.
      - `epoch_num`: Max number of epochs. Default is 600.
      - `cell_name`: Two cell can be choiced, including lstm, arnn.  Default is arnn.
      - `K`: Hyperparameter for the Att-LSTM, Default is 8.
      
- <b>Ploting Results</b>

  * All of test results are saved to the "result" folder, and each task has a corresponding folder. We just run the plot script: ``` python plot.py ```


### The Results
- <b>The results of RNNs on the adding problem </b>

|         |    |
| ------------- | -----:|
| ![](https://github.com/Tom957/Att-LSTM/blob/master/images/Add_T100.jpg)        | ![](https://github.com/Tom957/Att-LSTM/blob/master/images/Add_T200.jpg)   |
| ![](https://github.com/Tom957/Att-LSTM/blob/master/images/Add_T400.jpg)        | ![](https://github.com/Tom957/Att-LSTM/blob/master/images/Add_T600.jpg)   |

- <b>The results of LSTM and Att-LSTM on the pixel-by-pixel MNIST and pMNIST</b>

|         |    |
| ------------- | -----:|
| ![](https://github.com/Tom957/Att-LSTM/blob/master/images/MNIST.jpg)        | ![](https://github.com/Tom957/Att-LSTM/blob/master/images/pMNIST.jpg)   |


[1]:https://github.com/tensorflow/tensorflow
[2]:http://cogcomp.org/Data/QA/QC/
[3]:http://www.msmarco.org/dataset.aspx


