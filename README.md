## Novel Architecture for Long Short-Term Memory Used in Question Classification
In this paper, we propose a novel architecture (Att-LSTM) for basic LSTM that connects continuous hidden states of previous time steps to the current time step and applies an attention mechanism to these hidden states. This architecture can not only capture local features effectively but also help learn very long-distance correlations in an input sequence. We evaluate Att-LSTM in various sequential tasks, such as adding problem, sequence classification, and character-level language modeling. In addition, to
prove the generalization and practicality of the novel architecture, we design a character-level hierarchical Att-LSTM and refine the word representation with a highway network. This hierarchical model achieved excellent performance on question classification.

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


- <b> The accuracy on the TREC dataset</b> 
  * Test [01]: losses 0.013709, accuracys 0.566000
  * Test [02]: losses 0.009345, accuracys 0.762000
  * Test [03]: losses 0.005961, accuracys 0.836000
  * Test [04]: losses 0.003982, accuracys 0.884000
  * Test [05]: losses 0.003035, accuracys 0.904000
  * Test [06]: losses 0.002376, accuracys 0.918000
  * Test [07]: losses 0.002002, accuracys 0.928000
  * Test [08]: losses 0.001678, accuracys 0.940000
  * Test [09]: losses 0.001473, accuracys 0.954000
  * Test [10]: losses 0.001301, accuracys 0.958000
  * Test [11]: losses 0.001185, accuracys 0.958000
  * Test [12]: losses 0.001067, accuracys 0.960000
  * Test [13]: losses 0.000984, accuracys 0.968000
  * Test [14]: losses 0.000884, accuracys 0.970000
  * Test [15]: losses 0.000830, accuracys 0.972000
  * Test [16]: losses 0.000769, accuracys 0.974000
  * Test [17]: losses 0.000802, accuracys 0.972000
  * Test [18]: losses 0.000836, accuracys 0.970000
  * Test [19]: losses 0.000769, accuracys 0.972000
  * Test [20]: losses 0.000783, accuracys 0.972000
  * Test [21]: losses 0.000828, accuracys 0.970000
  * Test [22]: losses 0.000886, accuracys 0.968000
  * Test [23]: losses 0.000972, accuracys 0.962000
  * Test [24]: losses 0.000848, accuracys 0.970000
  * Test [25]: losses 0.000815, accuracys 0.968000
  * Test [26]: losses 0.000819, accuracys 0.968000
  * Test [27]: losses 0.000680, accuracys 0.980000
  * Test [28]: losses 0.000663, accuracys 0.982000
  * Test [29]: losses 0.000637, accuracys 0.980000
  * Test [30]: losses 0.000626, accuracys 0.980000
  * Test [31]: losses 0.000620, accuracys 0.980000
  * Test [32]: losses 0.000620, accuracys 0.980000
  * Test [33]: losses 0.000617, accuracys 0.980000
  * Test [34]: losses 0.000617, accuracys 0.982000
  * Test [35]: losses 0.000614, accuracys 0.980000
  * Test [36]: losses 0.000618, accuracys 0.982000
  * Test [37]: losses 0.000616, accuracys 0.980000
  * ......
  
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


