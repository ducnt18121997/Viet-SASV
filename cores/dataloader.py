import os
import glob
import numpy
import torch
import random
import librosa
import soundfile
import numpy as np
from scipy import signal
from typing import Optional, Tuple
from models.classifier import TrainingConfig
from setting import SETTING


class data_loader(object):
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config if config else TrainingConfig()

        # Load and configure augmentation files
        self.sample_rate = self.config.sampling_rate
        self.max_length = self.config.segment_length
        self.noisetypes = ["background", "speech", "music"]
        self.noisesnr = {"background": [0, 15], "speech": [13, 20], "music": [5, 15]}
        self.numnoise = {"background": [1, 1], "speech": [3, 8], "music": [1, 1]}

        self.noiselist = {
            _noise: glob.glob(os.path.join(SETTING.MUSAN_PATH, _noise, "*/*.wav"))
            for _noise in self.noisetypes
        }
        print("[*] Total noise sample: ")
        for k, v in self.noiselist.items():
            print(f"- {k}: {len(v)} samples")
        self.rir_files = glob.glob(os.path.join(SETTING.RIR_PATH, "*/*/*.wav"))
        print(f"- reverse: {len(self.rir_files)} samples")

        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(SETTING.TRAIN_LIST).read().splitlines()
        dictkeys = list(set([x.split("|")[1] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            file_name, speaker_label = line.split("|")
            speaker_label = dictkeys[speaker_label]
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
        self.use_augment = self.SETTING.USE_AUGMENT

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Read the utterance and randomly select the segment
        audio, sr = librosa.load(self.data_list[index], sr=self.sample_rate)
        if audio.shape[0] <= self.max_length:
            shortage = self.max_length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), "wrap")
        start_frame = numpy.int64(random.random() * (audio.shape[0] - self.max_length))
        audio = audio[start_frame : start_frame + self.max_length]
        audio = numpy.stack([audio], axis=0)

        # Data Augmentation
        augtype = random.randint(0, 4) if self.use_augment else 0
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, "speech")
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, "music")
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, "background")
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, "speech")
            audio = self.add_noise(audio, "music")
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self) -> int:
        return len(self.data_list)

    def add_rev(self, audio: np.ndarray) -> np.ndarray:
        rir_file = random.choice(self.rir_files)
        rir, sr = librosa.load(rir_file, sr=self.sample_rate)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode="full")[:, : self.max_length]

    def add_noise(self, audio: np.ndarray, noisecat: str) -> np.ndarray:
        clean_db = 10 * numpy.log10(numpy.mean(audio**2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            if noiseaudio.shape[0] <= self.max_length:
                shortage = self.max_length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), "wrap")
            start_frame = numpy.int64(
                random.random() * (noiseaudio.shape[0] - self.max_length)
            )
            noiseaudio = noiseaudio[start_frame : start_frame + self.max_length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio**2) + 1e-4)
            noisesnr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
            )
            noises.append(
                numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio
            )
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
