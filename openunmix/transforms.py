from typing import Optional

import torch
import torchaudio
from torch import Tensor
import torch.nn as nn

try:
    from asteroid_filterbanks.enc_dec import Encoder, Decoder
    from asteroid_filterbanks.transforms import to_torchaudio, from_torchaudio
    from asteroid_filterbanks import torch_stft_fb
except ImportError:
    pass


def make_filterbanks(n_fft=4096, n_hop=1024, center=False, sample_rate=44100.0, method="torch"):
    """
    这段程序是创建STFT（Short-Time Fourier Transform）和ISTFT（Inverse Short-Time Fourier Transform）算法的代码。
    首先，创建一个长度为n_fft的窗口函数。然后，根据提供的method参数的不同值，选择使用TorchSTFT或AsteroidSTFT作为STFT的实现。
    TorchSTFT的解码器则为TorchISTFT，而AsteroidSTFT的解码器则为AsteroidISTFT。
    最后，返回编码器和解码器两个对象。
    """
    # 创建一个长度为n_fft的窗口函数:本质上是窗口内进行加权求和(包含在Parameter内)
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    if method == "torch":
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "asteroid":
        fb = torch_stft_fb.TorchSTFTFB.from_torch_args(
            n_fft=n_fft,
            hop_length=n_hop,
            win_length=n_fft,
            window=window,
            center=center,
            sample_rate=sample_rate,
        )
        encoder = AsteroidSTFT(fb)
        decoder = AsteroidISTFT(fb)
    else:
        raise NotImplementedError
    return encoder, decoder


class AsteroidSTFT(nn.Module):
    def __init__(self, fb):
        super(AsteroidSTFT, self).__init__()
        self.enc = Encoder(fb)

    def forward(self, x):
        aux = self.enc(x)
        return to_torchaudio(aux)


class AsteroidISTFT(nn.Module):
    def __init__(self, fb):
        super(AsteroidISTFT, self).__init__()
        self.dec = Decoder(fb)

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        aux = from_torchaudio(X)
        return self.dec(aux, length=length)


class TorchSTFT(nn.Module):
    """Multichannel Short-Time-Fourier Forward transform
    uses hard coded hann_window.
    封装了一个傅里叶变化类,把时域信息转换成频域信息
    Args:
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        窗口长度(FFT在时间轴上的长度,以采样点数量表示)【FFT的长度N决定频域中的分辨率】
        【频域分别率=采样率/n_fft】
        n_hop (int, optional): transform hop size. Defaults to 1024.
        窗口步长(n_fft/4)
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
            如果为 True，信号第一个窗口将被零填充。 完美重建信号需要居中。 但是，在频谱图模型的训练过程中，它可以安全地关闭。默认为“true”
        window (nn.Parameter, optional): window function
    """

    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        window: Optional[nn.Parameter] = None,
    ):
        super(TorchSTFT, self).__init__()
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x: Tensor) -> Tensor:
        """STFT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
                nb_samples: batch_size
                nb_channels: 表示通道的数量
                nb_timesteps: 时间步的数量(每个通道的时间序列由`nb_timesteps`个时间步组成)
        Returns:
            STFT (Tensor): complex stft of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
                nb_samples: batch_size
                nb_channels: 双通道还是单通道
                nb_bins: 表示频率分析的频率范围内的频段的数量(表示梅普尔图的频率段的数量,每个频率段代表频率范围内的一段频率。)
                nb_frames:表示音频信号的分析时间段的数量，每个时间段都对应一个梅普尔图。
                2表示实部和虚部的数值(说明数组中的每一个元素代表的都是复数,每个复数由两个实数组成,默认为2)
                每个样本都是一个四维数组,代表多个通道的频谱数据,每个通道的频谱数据由多个频率分析得到的多个
                频率段组成,频率段的数量由`n_bin`参数指定.每个频率段的数据是在多个分析时间段`nb_frames`内得到的
                每个复数由两个实数组成,分别表示复数的实部和虚部
        """

        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        complex_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        # 该方法用于将复数的张量转换为实数的张量,有效地丢弃复数地虚部
        # 比如将1+2j经过此函数后变成1
        stft_f = torch.view_as_real(complex_stft)
        # unpack batch
        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])
        return stft_f


class TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """

    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        window: Optional[nn.Parameter] = None,
    ) -> None:
        super(TorchISTFT, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate

        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])

        y = torch.istft(
            torch.view_as_complex(X),
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )

        y = y.reshape(shape[:-3] + y.shape[-1:])

        return y


class ComplexNorm(nn.Module):
    r"""Compute the norm of complex tensor input.

    计算输入张量中每个复数的幅度(一个张量中的每个复数的实部和虚部都是张量的元素,代表了一个频率在某个时间点上的复杂振幅
    其中实部表示信号的振幅，虚部表示信号的相位),它会首先将其视为复数张量，然后取其模长，返回的输出形状取决于参数mono的值。
    如果mono为True，则将每个帧的振幅（magnitude）进行平均，以得到一个单通道的振幅表示，
    其形状为(nb_samples, nb_channels, 1, nb_frames)，以保留信号的总能量。
    如果mono为False，则输出形状为(nb_samples, nb_channels, nb_bins, nb_frames)。
    Extension of `torchaudio.functional.complex_norm` with mono


    Args:
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, mono: bool = False):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`    (nb_samples, nb_channels, nb_bins, nb_frames, 2)

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        # take the magnitude(振幅)
        # 将输入张量`spec`视为一个复数张量,并计算每个元素的模长(绝对值),即从复数域映射到实数域
        # 将复数张量转化为振幅(magnitude)张量,即取复数张量的模（magnitude），即复数的绝对值。
        # 这是为了将时频表示中的相位（phase）信息去除，仅保留能量信息。
        # 在语音处理中，这种处理常常被用于计算梅尔倒谱系数（Mel-frequency cepstral coefficients，MFCCs）等特征。
        spec = torch.abs(torch.view_as_complex(spec))

        # downmix in the mag domain to preserve energy
        # 将多通道的复杂谱图转化为单通道的振幅谱图，并保留总能量，以更好地保持信号质量。
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec
