- <b>The schematic of the Att-LSTM </b>  
![Model architecture](https://github.com/Tom957/Att-LSTM/blob/master/images/attlstm.jpg)

- <b> The schematic of the hierarchical Att-LSTM </b>
![Model architecture](https://github.com/Tom957/Att-LSTM/blob/master/images/hierarchical.jpg)


### Requirements
- Python 3.5.2
- [Tensorflow 0.12][1]

### Datasets
- Adding Problem was proposed by Hochreiter & Schmidhuber, and the dataset is create randomly. 
- A sequential version of the handwritten digit classification (MNIST), which is downloaded by tensorflow.
- [TREC][2] dataset that is the most common question classification.
- MSQC dataset is extracted from [Microsoft Research Question-Answering Corpus][3].


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
  * Question classification usage ``` python test_hierqc.py```
      - <b>Mian Options</b>
      - `batch_size`: Batch Size. Default is 100.
      - `data_type`: Two dataset be choiced, including trec, msqc. Default is trec
      - `char_embed_dim`: Dimension of input. Default is 20.
      - `first_unit_size`:  The number of low-level Att-LSTM hidden unit. Default is 40.
      - `secod_unit_size`:  The number of high-level Att-LSTM hidden unit. Default is 40.
      - `learning_rate`: Learning Rate. Default is 0.0001.
      - `clip_norm`: Clip gradient and avoid gradient explosion. Default is 10.0.
      - `epoch_num`: Max number of epochs. Default is 300.
      - `cell_name`: Two cell can be choiced, including lstm, arnn.  Default is arnn.
      - `K`: Hyperparameter for the Att-LSTM, Default is 2.
      
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

  
- <b> The accuracy on the MSQC dataset</b> 
  * Test [01]: losses 0.003270, accuracys 0.893542
  * Test [02]: losses 0.002582, accuracys 0.910729
  * Test [03]: losses 0.002295, accuracys 0.920417
  * Test [04]: losses 0.002140, accuracys 0.923333
  * Test [05]: losses 0.002080, accuracys 0.926875
  * Test [06]: losses 0.002057, accuracys 0.926354
  * Test [07]: losses 0.002051, accuracys 0.927083
  * Test [08]: losses 0.002086, accuracys 0.926875
  * Test [09]: losses 0.002123, accuracys 0.928229
  * Test [10]: losses 0.002205, accuracys 0.928438
  * Test [11]: losses 0.002328, accuracys 0.927500
  * Test [12]: losses 0.002507, accuracys 0.926146
  * Test [13]: losses 0.002532, accuracys 0.925000
  * Test [14]: losses 0.002589, accuracys 0.926146
  * Test [15]: losses 0.002810, accuracys 0.921875
  * Test [16]: losses 0.003093, accuracys 0.917500
  * Test [17]: losses 0.003234, accuracys 0.920208
  * Test [18]: losses 0.003134, accuracys 0.922917
  * Test [19]: losses 0.003107, accuracys 0.928854
  * Test [20]: losses 0.003103, accuracys 0.930938
  * Test [21]: losses 0.003237, accuracys 0.930417
  * Test [22]: losses 0.003370, accuracys 0.933646
  * Test [23]: losses 0.003573, accuracys 0.932188
  * Test [24]: losses 0.003570, accuracys 0.929583
  * Test [25]: losses 0.003672, accuracys 0.930208
  * Test [26]: losses 0.003768, accuracys 0.930000
  * Test [27]: losses 0.003883, accuracys 0.929688
  * Test [28]: losses 0.003972, accuracys 0.928958
  * Test [29]: losses 0.003936, accuracys 0.931458
  * Test [30]: losses 0.003931, accuracys 0.931979
  * Test [31]: losses 0.003995, accuracys 0.931563
  * Test [32]: losses 0.004023, accuracys 0.930208
  * Test [33]: losses 0.004210, accuracys 0.925417
  * Test [34]: losses 0.004104, accuracys 0.929792
  * Test [35]: losses 0.004203, accuracys 0.928958
  * Test [36]: losses 0.004165, accuracys 0.932500
  * Test [37]: losses 0.004214, accuracys 0.929792
  * ......

  
[1]:https://github.com/tensorflow/tensorflow
[2]:http://cogcomp.org/Data/QA/QC/
[3]:http://www.msmarco.org/dataset.aspx


