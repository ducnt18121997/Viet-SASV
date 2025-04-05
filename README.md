<div align="center">

# End-to-end Speaker Verification (/w Spoof Aware) - PyTorch Implementation

A PyTorch-based implementation of **Speaker Verification** utilizing deep learning architectures and language models, with support for speaker recognition on the [Vietnamese dataset](https://github.com/datvithanh/vietnamese-sv-dataset) by Dean Nguyen.
</div>

## Table of Contents
- [Architecture](#architecture)
- [Augmentation](#augmentation)
- [Setup](#setup)
- [Running](#running)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)
- [Responsibility](#responsibility)

## Architecture

- [x] [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143) (Desplanques et al., 2020)
- [x] [Analysis of Score Normalization in Multilingual Speaker Recognition](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/0803.PDF) (Matejka et al., 2017)
- [x] [Speaker Recognition from raw waveform with SincNet](https://arxiv.org/abs/1808.00158) (Ravanelli & Bengio, 2018)
- [x] [GE2E Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) (Wan et al., 2018)
- [x] [AAM-Softmax Loss for Speaker Verification](https://arxiv.org/abs/1801.07698) (Deng et al., 2018)

## Augmentation

- [x] [MUSAN: A Music, Speech, and Noise Corpus](https://arxiv.org/abs/1510.08484) (Snyder et al., 2015)
- [x] [Room Impulse Response and Noise Database](https://www.openslr.org/28/) (Ko et al., 2017)
- [x] [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) (Park et al., 2019)

## Setup

```bash
conda create --name venv python=3.8.10
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Running

Change settings in `setting/setting.yaml` and run:

```bash
python cores/train.py
```

## References
> NOTE: This project was developed quite a while ago, so I may have missed some information about the repositories used. Please create a ticket so I can add them here.

- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [ASSIST](https://github.com/clovaai/aasist)
- [SincNet](https://github.com/mravanelli/SincNet)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Responsibility

This implementation is provided as-is, without any warranties or guarantees. The authors are not responsible for any misuse or damage caused by this software. Users are responsible for:

1. Ensuring proper data privacy and security when using this software
2. Complying with all applicable laws and regulations
3. Obtaining necessary permissions for any data used
4. Properly citing and acknowledging the original authors of the referenced papers
5. Understanding and accepting the limitations of the models and algorithms

The implementation is based on academic research papers and should be used for research purposes only. Commercial use may require additional permissions and compliance with relevant regulations.

## Citation
If you use this code in your research, please cite:
```bibtex
@misc{deanng_2025,
    author = {Dean Nguyen},
    title = {End-to-end Speaker Verification - PyTorch Implementation},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ducnt18121997/Viet-SASV}}
}
```