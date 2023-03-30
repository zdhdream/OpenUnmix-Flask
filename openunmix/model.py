from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        频率分量的数量
        通常越高的nb_bins会提供更准确的频率分辨率，但也需要更多的计算资源和存储空间。
        在实际应用中，nb_bins的具体取值需要根据实际情况进行选择。
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
            self,
            nb_bins: int = 4096,
            nb_channels: int = 2,
            hidden_size: int = 512,
            nb_layers: int = 3,
            unidirectional: bool = False,
            input_mean: Optional[np.ndarray] = None,
            input_scale: Optional[np.ndarray] = None,
            max_bin: Optional[int] = None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        # inputs:(16, 2, 2049, 255)
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)  # (255, 16, 2, 2049)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        # 从输入的频谱图`x`中取出最后一维度的前`self.nb_bins`个数,即将最后一维数的数量限制在self.nb_bins
        # 这个操作是对输入的特征图数据进行裁剪的意义。特征图中的每一帧是频率分量的集合，而频率分量数目可能不同。
        # 因此在该代码中使用了变量`self.nb_bins`作为裁剪的阈值，只保留前`self.nb_bins`个频率分量的信息
        # 裁剪的目的是什么呢？频率分量数量过多，不仅会增加计算复杂度，也有可能会导致模型对于噪音信息过于敏感。
        # 所以，通常情况下，为了提高模型的鲁棒性，在训练过程中会对频率分量数量进行一定的限制，从而减少计算量。
        # 这个操作的实施不能保证模型训练的好坏，因为这个操作本身并不直接影响模型的训练。但是，它可以减少计算量，帮助模型训练得更快。
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        # 对输入数据进行归一化处理
        x = x + self.input_mean
        x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))  # (255, 16, 512)
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)  # (255, 16, 512)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)  # [0](255, 16, 512)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)  # (255, 16, 512)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))  # (4080, 512)
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)  # (255, 16, 2, 2049)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.
    这是一个用于语音源分离的Pytorch模型,即从混合语音中分离出不同的音频源
    该模型接受一个混合语音的批处理作为输入，输出每个音频源的音频波形。它可以使用多种不同的目标模型来执行分离
    在训练时，这个模型通过反向传播优化目标模型。以便更好地分离混合语音。在推理时，目标模型将用于分离混合语音。
    此模型还支持Wiener过滤和EM迭代,以提高分离性能
    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
            EM 迭代次数,在后处理阶段用于细化初始估计。如果只估计一个目标，则设置为0，在任何潜在的EM后处理之前，默认为`False`
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
        softmask(bool): softmask参数用于指定是否使用软掩码来计算词嵌入。
        当softmask=True时，掩码不会被处理为0向量，而是采用了一个非常小的值来代替填充的部分，
        这允许在使用掩码时保留一些语义信息(保留原始向量的信息)，同时仍然将掩码应用于词嵌入的计算。这在一些应用中可能很有用，
        例如在对序列进行预测时，需要同时考虑输入序列和掩码的信息。
        我们在计算模型的损失函数时，被填充的向量所对应的位置不会被考虑在内，也不会对模型的训练产生影响
        这样可以避免填充向量对模型训练造成的负面影响
    """

    def __init__(
            self,
            target_models: Mapping[str, nn.Module],
            niter: int = 0,
            softmask: bool = False,
            residual: bool = False,
            sample_rate: float = 44100.0,
            n_fft: int = 4096,
            n_hop: int = 1024,
            nb_channels: int = 2,
            wiener_win_len: Optional[int] = 300,
            filterbank: str = "torch",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`

        `nb_frames`: 每个时间窗口的帧数(`nb_timesteps`: 被分割成多个时间窗口，并对每个时间窗口进行FFT)
        `nb_bins`: 表示频率的数量（即FFT的大小一半）
        `2`: 表示实部和虚部构成的复数
        """

        nb_sources = self.nb_targets  # 目标源
        nb_samples = audio.shape[0]  # 样本数

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)  # 将混合音频进行傅里叶变换得到时频图(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        X = self.complexnorm(mix_stft)  # (nb_samples, nb_channels, nb_bins, nb_frames)。    # 去除相位信息，保留能量信息

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype,
                                   device=X.device)  # `(nb_samples, nb_channels, nb_bins, nb_frames, nb_sources)` 包含了每个音源在每个时频单元上的幅度谱信息

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        # 这段代码的作用是将形状为 (nb_samples, nb_channels, nb_bins, nb_frames, 2) 的混合信号 STFT 转置为形状为 (nb_samples, nb_frames, nb_bins, nb_channels, 2) 的 STFT，以便于后续的滤波处理。
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )   # (nb_samples, nb_channels, nb_bins, nb_frames, 2, nb_sources )
        # 计算目标音频信号的 STFT ，对于每个样本，通过循环遍历音频的所有帧来计算目标音频信号的 STFT。
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                # 确定当前需要处理的音频帧的索引。
                # `pos`: 当前音频帧的起始位置，`wiener_win_len`: 窗口大小
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1
                #
                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],    # 混合信号 STFT 对应的谱图
                    mix_stft[sample, cur_frame],    # 当前帧的混合信号
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )   # (nb_samples, nb_channels, nb_bins, nb_frames, 2, nb_sources)即每个样本的目标音频信号 STFT 在最后一个维度上排列。

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = self.istft(targets_stft, length=audio.shape[2])

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
