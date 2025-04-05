import sys
import tqdm
import time
import math
import librosa
import numpy as np
from typing import Literal, Union, Tuple, Optional, Dict
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.helper import tune_threshold_from_score, compute_error_rates, compute_min_dcf


class PreEmphasis(nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class TrainingConfig(BaseModel):
    # Data configuration
    sampling_rate: int = 16000
    max_wav_value: float = 32768.0
    segment_length: int = 32768
    # STFT configuration
    filter_length: int = 512  # fft points
    hop_length: int = 160  # hop size
    win_length: int = 400  # window size
    window: str = "hann"  # window type
    # Mel configuration
    mel_channels: int = 80  # number of Mel basis
    mel_fmin: int = 20  # minimum frequency for mel basis
    mel_fmax: int = 7600  # maximum frequency for Mel basis
    log_base: None
    num_frames: int = 200
    # Training configuration
    loss_func: Literal["aam_softmax", "softmax", "nnnloss"] = "aam_softmax"
    loss_margin: float = 0.2  # Loss margin in AAM softmax
    loss_scale: int = 30  # Loss scale in AAM softmax
    optim_lr: float = 0.01  # Learning rate
    optim_wd: float = 2.0e-5  # Weight decay
    optim_lr_decay: float = 0.98  # Learning rate decay every [test_step] epoch(s)


class SpeechClassifier(nn.Module):
    def __init__(
        self,
        n_class: int,
        config: Optional[TrainingConfig] = None,
        frontend_version: Literal["mel_spectrogram", "sinc_net"] = "mel_spectrogram",
        frontend_config: Optional[Dict] = None,
        backend_version: Literal["xvector", "ecapa_tdnn", "assist"] = "ecapa_tdnn",
        backend_config: Optional[Dict] = None,
    ):
        super(SpeechClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config if config else TrainingConfig()
        self.sr = self.config.sampling_rate
        self.max_length = self.config.num_frames * 160 + 240

        ## Initialize frontend model
        self.frontend_version = frontend_version
        if self.frontend_version == "mel_spectrogram":
            from torchaudio.transforms import MelSpectrogram

            self.feat_extractor = nn.Sequential(
                PreEmphasis(),
                MelSpectrogram(
                    sample_rate=self.config.sampling_rate,
                    n_fft=self.config.filter_length,
                    win_length=self.config.win_length,
                    hop_length=self.config.hop_length,
                    f_max=self.config.mel_fmax,
                    f_min=self.config.mel_fmin,
                    window_fn=torch.hamming_window,
                    n_mels=self.config.mel_channels,
                ),
            )
            self.feat_dims = self.config.mel_channels
        elif self.frontend_version == "sinc_net":
            from models.frontend.sinc_net import SincNet
            from models.frontend.sinc_net import SincNetConfig

            self.fs = self.config.sampling_rate
            self.feat_extractor = SincNet(
                sample_rate=self.fs,
                input_dim=self.max_length,
                config=SincNetConfig(**frontend_config) if frontend_config else None,
            )
            self.feat_dims = self.feat_extractor.out_dim
        else:
            raise NotImplementedError(
                f"Not implemented frontend {self.frontend_version} yet!"
            )
        self.feat_extractor.to(self.device)

        ## Initialize backend model
        self.backend_version = backend_version
        if self.backend_version == "xvector":
            from models.backend.xvector import XVector
            from models.backend.xvector import XVectorConfig

            self.speaker_encoder = XVector(
                idims=self.feat_dims,
                config=XVectorConfig(**backend_config) if backend_config else None,
            )
        elif self.backend_version == "ecapa_tdnn":
            from models.backend.tdnn import ECAPA_TDNN
            from models.backend.tdnn import ECAPA_TDNNConfig

            self.speaker_encoder = ECAPA_TDNN(
                idims=self.feat_dims,
                config=ECAPA_TDNNConfig(**backend_config) if backend_config else None,
            )
        elif self.backend_version == "assist":
            from models.backend.assist import Assist
            from models.backend.assist import AssistConfig

            self.speaker_encoder = Assist(
                config=AssistConfig(**backend_config) if backend_config else None,
            )
            if self.frontend_version == "sinc_net":
                # Flatten the output for frontend model when using sinc_net
                self.feat_extractor.flatten = True
        else:
            raise NotImplementedError(
                f"Not implemented backend {self.backend_version} yet!"
            )
        self.speaker_encoder.to(self.device)

        ## Initialize loss function
        self.loss_func = self.config.loss_func
        if self.loss_func == "aam_softmax":
            from models.loss import AAMsoftmax

            self.speaker_loss = AAMsoftmax(
                nClasses=n_class,
                nOut=self.speaker_encoder.odims,
                margin=self.config.loss_margin,
                scale=self.config.loss_scale,
            )
        elif self.loss_func == "softmax":
            from models.loss import Softmax

            self.speaker_loss = Softmax(
                nOut=self.speaker_encoder.odims,
                nClasses=n_class,
            )
        elif self.loss_func == "nnnloss":
            from models.loss import NNNLoss

            self.speaker_loss = NNNLoss(
                nOut=self.speaker_encoder.odims,
                nClasses=n_class,
            )
        else:
            raise NotImplementedError(
                f"Not implemented loss function {self.loss_func} yet!"
            )

        ## Initialize optimizer & scheduler
        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.config.optim_lr,
            weight_decay=self.config.optim_wd,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=1,
            gamma=self.config.optim_lr_decay,
        )

        ## Statistics parameters
        self.n_params = sum(param.numel() for param in self.parameters())
        print(
            time.strftime("%m-%d %H:%M:%S")
            + f" Model parameter number {round(self.n_params / 1024 / 1024, 2)}M && {n_class} speakers"
        )

    def _forward(self, datas: torch.Tensor, aug: bool = False) -> torch.Tensor:
        if self.frontend_version == "mel_spectrogram":
            feats = self.feat_extractor(datas) + 1e-6
            feats = feats.log()
            feats = feats - torch.mean(
                feats, dim=-1, keepdim=True
            )  # NOTE: input normalization
        elif self.frontend_version.startswith("sinc_net"):
            feats = self.feat_extractor(datas)

        embeds = self.speaker_encoder(feats, aug=aug)
        return embeds

    def _extract_feature(
        self, audio_data: Union[str, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(audio_data, str):
            audio, _ = librosa.load(audio_data, sr=self.sr)
        else:
            audio = audio_data

        # full utterance
        if self.frontend_version == "mel_spectrogram":
            data_1 = torch.FloatTensor(np.stack([audio], axis=0)).to(self.device)
        else:
            bs = math.ceil(audio.shape[0] / self.max_length)
            data_1 = torch.zeros(bs, self.max_length)
            i, s, e = 0, 0, self.max_length
            while s < audio.shape[0]:
                _data = torch.FloatTensor(audio[s:e])
                data_1[i][: _data.size(0)] = _data
                i += 1
                s += self.max_length
                e += self.max_length
            data_1 = data_1.to(self.device)

        # splited utterance
        if audio.shape[0] <= self.max_length:
            shortage = self.max_length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), "wrap")
        feats = []
        startframe = np.linspace(0, audio.shape[0] - self.max_length, num=5)
        for asf in startframe:
            feats.append(audio[int(asf) : int(asf) + self.max_length])
        feats = np.stack(feats, axis=0).astype(np.float)
        data_2 = torch.FloatTensor(feats).to(self.device)

        return (data_1, data_2)

    def train_network(
        self, epoch: int, loader: DataLoader
    ) -> Tuple[float, float, float]:
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optimizer.param_groups[0]["lr"]
        for num, (data, labels) in enumerate(loader, start=1):
            self.optimizer.zero_grad()
            labels = torch.LongTensor(labels).to(self.device)
            speaker_embedding = self._forward(datas=data.to(self.device), aug=True)
            nloss, prec = self.speaker_loss(speaker_embedding, labels)
            nloss.backward()
            self.optimizer.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(
                time.strftime("%m-%d %H:%M:%S")
                + " [%2d] Lr: %5f, Training: %.2f%%, "
                % (epoch, lr, 100 * (num / loader.__len__()))
                + " Loss: %.5f, ACC: %2.2f%% \r"
                % (loss / (num), top1 / index * len(labels))
            )
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list: str) -> Tuple[float, float]:
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[0])
            files.append(line.split()[1])
        setfiles = list(set(files))
        setfiles.sort()

        for file in tqdm.tqdm(setfiles, total=len(setfiles)):
            data_1, data_2 = self._extract_feature(file)

            # speaker embeddings
            with torch.no_grad():
                embedding_1 = self._forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1).mean(
                    dim=0, keepdim=True
                )
                embedding_2 = self._forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1).mean(
                    dim=0, keepdim=True
                )
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]

            # compute the scores
            score_1 = torch.mean(
                torch.matmul(embedding_11, embedding_21.T)
            )  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[2]))

        # Coumpute EER and minDCF
        EER = tune_threshold_from_score(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        minDCF, _ = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def eval_network_with_asnorm(
        self, eval_list: str, cohort_embs: np.array
    ) -> Tuple[float, float]:
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[0])
            files.append(line.split()[1])
        setfiles = list(set(files))
        setfiles.sort()

        for _, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            data_1, data_2 = self._extract_feature(file)
            # speaker embeddings
            with torch.no_grad():
                embedding_1 = self._forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self._forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1).mean(
                    dim=0, keepdim=True
                )
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        for line in tqdm.tqdm(lines, desc="scoring"):
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]
            # compute the scores
            score_1 = (
                as_norm(
                    torch.mean(torch.matmul(embedding_11, embedding_21.T)),
                    embedding_11,
                    embedding_21,
                    cohort_embs[0],
                )
                .detach()
                .cpu()
                .numpy()
            )

            score_2 = (
                as_norm(
                    torch.mean(torch.matmul(embedding_12, embedding_22.T)),
                    embedding_12,
                    embedding_22,
                    cohort_embs[1],
                )
                .detach()
                .cpu()
                .numpy()
            )

            score = (score_1 + score_2) / 2
            scores.append(score)
            labels.append(int(line.split()[2]))

        # Coumpute EER and minDCF
        EER = tune_threshold_from_score(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = compute_error_rates(scores, labels)
        minDCF, _ = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def build_cohort(self, train_list: DataLoader) -> list[torch.Tensor]:
        self.eval()
        embeddings = [[], []]
        for filename in tqdm.tqdm(train_list, total=len(train_list)):
            data_1, data_2 = self._extract_feature(filename)

            # speaker embeddings
            with torch.no_grad():
                embedding_1 = self._forward(data_1, aug=False)
                embeddings[0].append(F.normalize(embedding_1, p=2, dim=1))
                embedding_2 = self._forward(data_2, aug=False)
                embeddings[1].append(
                    F.normalize(embedding_2, p=2, dim=1).mean(dim=0, keepdim=True)
                )

        return [torch.stack(x) for x in embeddings]

    def save_parameters(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_parameters(self, path: str) -> None:
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, self_state[name].size(), loaded_state[origname].size())
                )
                continue
            self_state[name].copy_(param)


def as_norm(
    score: torch.Tensor,
    embedding_1: torch.Tensor,
    embedding_2: torch.Tensor,
    cohort_feats: torch.Tensor,
    topk: int = 1000,
) -> torch.Tensor:
    """NOTE(by deanng): This function is only use for ASNorm loss function.
    Which only support to increase the score in testing phase.
    """
    score_1 = torch.matmul(cohort_feats, embedding_1.T)[:, 0]
    score_1 = torch.topk(score_1, topk, dim=0)[0]
    mean_1 = torch.mean(score_1, dim=0)
    std_1 = torch.std(score_1, dim=0)

    score_2 = torch.matmul(cohort_feats, embedding_2.T)[:, 0]
    score_2 = torch.topk(score_2, topk, dim=0)[0]
    mean_2 = torch.mean(score_2, dim=0)
    std_2 = torch.std(score_2, dim=0)

    score = 0.5 * (score - mean_1) / std_1 + 0.5 * (score - mean_2) / std_2
    return score
