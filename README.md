# Zero Pronoun Resolution with Attention-based Neural Network
Recent neural network methods for zero pronoun resolution explore multiple models for generating representation vectors for zero pronouns and their candidate antecedents. Typically, contextual information is utilized to encode the zero pronouns since they are simply gaps that contain no actual content. To better utilize contexts of the zero pronouns, we here introduce the self-attention mechanism for encoding zero pronouns. With the help of the multiple hops of attention, our model is able to focus on some informative parts of the associated texts and therefore produces an efficient way of encoding the zero pronouns. In addition, an attention-based recurrent neural network is proposed for encoding candidate antecedents by their contents. Experiment results are encouraging: our proposed attention-based model gains the best performance on the Chinese portion of the OntoNotes corpus, substantially surpasses existing Chinese zero pronoun resolution baseline systems.


## Requirements
* Python 2.7
   * Pytorch(0.4.0)
   * CUDA

## Training Instructions
* Experiment configurations could be found in `conf.py`
* Run `./setup.sh` # it builds the data for training and testing from Ontonotes data.
    * It unzip data from `./data/zp_raw_data.zip` and store it in `./data/zp_data`
    * It devides the training dataset into the training and develpment set. The dataset is stored as `train_data` 
* Run `./start.sh` # train the model and get results (about 57.3% in F-score) .
    *   It takes about 3 minutes for one epoch

## Other Quirks
* It does use GPUs by default. Please make sure that the GPUs are vailable.
    * The default device utilized is `gpu0`, to use other GPUs, please add `-gpu $DEVICE_NUMBER` to the script `start.sh` after `main.py`.
