# MOSNet
Implementation of  "MOSNet: Deep Learning based Objective Assessment for Voice Conversion"
https://arxiv.org/abs/1904.08352

## Dependency
Linux Ubuntu 16.04
- GPU: GeForce RTX 2080 Ti
- Driver version: 418.67
- CUDA version: 10.1

Python 3.5
- tensorflow-gpu==2.0.0-beta1 (cudnn=7.6.0)
- scipy
- pandas
- matplotlib
- librosa

### Environment set-up
For example,
```
conda create -n mosnet python=3.5
conda activate mosnet
pip install -r requirements.txt
conda install cudnn=7.6.0
```

## Usage

### Reproducing results in the paper

1. `cd ./data` and run `bash download.sh` to download the VCC2018 evaluation results and submitted speech. (downsample the submitted speech might take some times)
2. Run `python mos_results_preprocess.py` to prepare the evaluation results. (Run `python bootsrap_estimation.py` to do the bootstrap experiment for intrinsic MOS calculation)
3. Run `python utils.py` to extract .wav to .h5
4. Run `python train.py --model CNN-BLSTM` to train a CNN-BLSTM version of MOSNet. ('CNN', 'BLSTM' or 'CNN-BLSTM' are supported in model.py, as described in paper)
5. Run `python test.py` to test on the pre-trained weights with specified model and weight.


#### Note
The experimental results showed in the paper were trained on Keras with tensorflow 1.4.1 backend. However, the implementation here is based on tf2.0.0b1, so the results might vary a little. Additionally, the architectures showed in the paper were meta-architectures, any replace CNN/BLSTM with more fancy modules (ResNet etc.) would improve the final results. Tuning the hyper-parameters might result in the same favour. 

### Evaluating your custom waveform samples

1. Put the waveforms you wish to evaluate in a folder. For example, `<path>/<to>/<samples>`
2. Run `python python ./custom_test.py --rootdir <path>/<to>/<samples>`

This script will evaluate all the `.wav` files in `<path>/<to>/<samples>`, and write the results to `<path>/<to>/<samples>/MOSnet_result_raw.txt`. By default, the `pre_trained/cnn_blstm.h5` pretrained model is used. If you wish to use other models, please specify a different `--pretrained_model` and also change `from model import <model_to_be_used>`.

## Citation

If you find this work useful in your research, please consider citing:
```
@inproceedings{mosnet,
  author={Lo, Chen-Chou and Fu, Szu-Wei and Huang, Wen-Chin and Wang, Xin and Yamagishi, Junichi and Tsao, Yu and Wang, Hsin-Min},
  title={MOSNet: Deep Learning based Objective Assessment for Voice Conversion},
  year=2019,
  booktitle={Proc. Interspeech 2019},
}
```
 
 
## License

This work is released under MIT License (see LICENSE file for details).


## VCC2018 Database & Results

The model is trained on the large listening evaluation results released by the Voice Conversion Challenge 2018.<br>
The listening test results can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3257)<br>
The databases and results (submitted speech) can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3061)<br>
