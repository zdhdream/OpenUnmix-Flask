from typing import Optional, Union

import torch
import os
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json
import openunmix
# import openunmix
from openunmix import model


def bandwidth_to_max_bin(rate: float, n_fft: int, bandwidth: float) -> np.ndarray:
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    该函数通过计算得到一组频率分量(实现了将带宽转换为最大频率分量的函数)
    得到一组频率分量通常是指在音频信号处理中，使用特定算法将原始的时域信号转化为频域信号，得到一组频率分量。通常使用傅里叶变换（FFT）或者其他频域变换算法实现。
    每一个频率分量代表了原始信号中的某一个频率的幅值。这些频率分量可以更直观的反映出信号的频率分布情况，从而有助于更好的频域特征提取、分析和处理。
    Args:
        rate (int): Sample rate
        n_fft (int): FFT length
        bandwidth (float): Target bandwidth in Hz

    Returns:
        np.ndarray: maximum frequency bin  最大频率分量
    """
    # 从0到采样率的一半均匀分配了`n_fft//2+1`个频率点,其中endpoint表示包含最后一个样本点
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1, endpoint=True)
    # 找到所有频率分量小于等于目标带宽的索引，并使用 Numpy 库的 max 函数求出索引的最大值，最后加上 1，作为该函数的返回值。
    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(state: dict, is_best: bool, path: str, target: str):
    """Convert bandwidth to maximum bin count
    用于将训练过程中的模型参数保存到硬盘
    Assuming lapped transforms such as STFT

    Args:
        state (dict): torch model state dict
        字典，包含当前训练状态的信息，包括模型参数、优化器状态等。
        is_best (bool): if current model is about to be saved as best model
        一个布尔值，表示当前的模型是否是最好的模型。
        path (str): model path
        一个字符串，表示保存模型的路径。
        target (str): target name
        target：一个字符串，表示保存模型的名称。
    """
    # save full checkpoint including optimizer
    torch.save(state, os.path.join(path, target + ".chkpnt"))
    if is_best:
        # save just the weights
        torch.save(state["state_dict"], os.path.join(path, target + ".pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 值的总和
        self.count = 0  # 值的数量

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """Early Stopping Monitor

    该类实现了一种提前终止训练的监测机制
    具体地,在训练过程中,每一轮都会使用一个评价指标对模型地表现进行评估,如果评价指标的值在连续的几轮迭代中没有改善
    则认为训练已经不再有效,需要提前终止训练。
    指标的增量是通过参数`min_delta`确定的,如果评价指标的值在每轮迭代中都没有提高超过这个阈值,则认为该模型已经不再有改进的余地。
    如果在一定的连续轮数内(由`patience`参数确定)评价指标的值都没有改善,则算法会提前终止训练
    参数`mode`可以设置为`min`或者`max`，表示评价指标是最小值还是最大值。
    这段代码实现了提前终止训练的监测机制，可以有效的避免过拟合，从而提高模型的泛化能力。
    """

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        """
        这段程序的意思是：每次调用 step 方法时，都会对模型的性能进行评估，
        具体的评估方法是比较当前的性能指标 metrics 和目前最佳性能指标 best 的大小关系。
        如果当前性能指标比目前最佳性能指标更优，那么把当前性能指标设置为最佳性能指标；
        如果当前性能指标不比目前最佳性能指标更优，那么将不良的次数加一。如果不良的次数超过了预先设置的 patience 值，说明模型的性能开始下降，此时函数返回 True。否则，函数返回 False。
        """
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        """
            这段代码的作用是初始化一个比较函数 is_better，用于比较当前的指标值与最优指标值的大小关系。
            如果 mode 为 "min"，则 is_better 函数判断当前指标值 a 是否小于最优指标值 best 减去参数 min_delta，即是否比最优指标值更优
            如果 mode 为 "max"，则 is_better 函数判断当前指标值 a 是否大于最优指标值 best 加上参数 min_delta，即是否比最优指标值更优
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


def load_target_models(targets, model_str_or_path="umxhq", device="cpu", pretrained=True):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
    if isinstance(targets, str):
        targets = [targets]

    model_path = Path(model_str_or_path).expanduser()
    if not model_path.exists():
        # model path does not exist, use pretrained models
        try:
            # disable progress bar
            hub_loader = getattr(openunmix, model_str_or_path + "_spec")
            err = io.StringIO()
            with redirect_stderr(err):
                return hub_loader(targets=targets, device=device, pretrained=pretrained)
            print(err.getvalue())
        except AttributeError:
            raise NameError("Model does not exist on torchhub")
            # assume model is a path to a local model_str_or_path directory
    else:
        models = {}
        for target in targets:
            # load model from disk
            with open(Path(model_path, target + ".json"), "r") as stream:
                results = json.load(stream)

            target_model_path = next(Path(model_path).glob("%s*.pth" % target))
            state = torch.load(target_model_path, map_location=device)

            models[target] = model.OpenUnmix(
                nb_bins=results["args"]["nfft"] // 2 + 1,
                nb_channels=results["args"]["nb_channels"],
                hidden_size=results["args"]["hidden_size"],
                max_bin=state["input_mean"].shape[0],
            )

            if pretrained:
                models[target].load_state_dict(state, strict=False)

            models[target].to(device)
        return models


def load_separator(
        model_str_or_path: str = "umxhq",
        targets: Optional[list] = None,
        niter: int = 1,
        residual: bool = False,
        wiener_win_len: Optional[int] = 300,
        device: Union[str, torch.device] = "cpu",
        pretrained: bool = True,
        filterbank: str = "torch",
):
    """Separator loader

    Args:
        model_str_or_path (str): Model name or path to model _parent_ directory
            E.g. The following files are assumed to present when
            loading `model_str_or_path='mymodel', targets=['vocals']`
            'mymodel/separator.json', mymodel/vocals.pth', 'mymodel/vocals.json'.
            Defaults to `umxhq`.
        targets (list of str or None): list of target names. When loading a
            pre-trained model, all `targets` can be None as all targets
            will be loaded
        niter (int): Number of EM steps for refining initial estimates
            in a post-processing stage. `--niter 0` skips this step altogether
            (and thus makes separation significantly faster) More iterations
            can get better interference reduction at the price of artifacts.
            Defaults to `1`.
        residual (bool): Computes a residual target, for custom separation
            scenarios when not all targets are available (at the expense
            of slightly less performance). E.g vocal/accompaniment
            Defaults to `False`.
        wiener_win_len (int): The size of the excerpts (number of frames) on
            which to apply filtering independently. This means assuming
            time varying stereo models and localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
            Defaults to `300`
        device (str): torch device, defaults to `cpu`
        pretrained (bool): determines if loading pre-trained weights
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """
    model_path = Path(model_str_or_path).expanduser()

    # when path exists, we assume its a custom model saved locally
    if model_path.exists():
        if targets is None:
            raise UserWarning("For custom models, please specify the targets")

        target_models = load_target_models(
            targets=targets, model_str_or_path=model_path, pretrained=pretrained
        )

        with open(Path(model_path, "separator.json"), "r") as stream:
            enc_conf = json.load(stream)

        separator = model.Separator(
            target_models=target_models,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            sample_rate=enc_conf["sample_rate"],
            n_fft=enc_conf["nfft"],
            n_hop=enc_conf["nhop"],
            nb_channels=enc_conf["nb_channels"],
            filterbank=filterbank,
        ).to(device)

    # otherwise we load the separator from torchhub
    else:
        hub_loader = getattr(openunmix, model_str_or_path)
        separator = hub_loader(
            targets=targets,
            device=device,
            pretrained=True,
            niter=niter,
            residual=residual,
            filterbank=filterbank,
        )

    return separator


def preprocess(
        audio: torch.Tensor,
        rate: Optional[float] = None,
        model_rate: Optional[float] = None,
) -> torch.Tensor:
    """
    From an input tensor, convert it to a tensor of shape
    shape=(nb_samples, nb_channels, nb_timesteps). This includes:
    -  if input is 1D, adding the samples and channels dimensions.
    -  if input is 2D
        o and the smallest dimension is 1 or 2, adding the samples one.
        o and all dimensions are > 2, assuming the smallest is the samples
          one, and adding the channel one
    - at the end, if the number of channels is greater than the number
      of time steps, swap those two.
    - resampling to target rate if necessary

    Args:
        audio (Tensor): input waveform
        rate (float): sample rate for the audio
        model_rate (float): sample rate for the model

    Returns:
        Tensor: [shape=(nb_samples, nb_channels=2, nb_timesteps)]
    """
    shape = torch.as_tensor(audio.shape, device=audio.device)

    if len(shape) == 1:
        # assuming only time dimension is provided.
        audio = audio[None, None, ...]
    elif len(shape) == 2:
        if shape.min() <= 2:
            # assuming sample dimension is missing
            audio = audio[None, ...]
        else:
            # assuming channel dimension is missing
            audio = audio[:, None, ...]
    if audio.shape[1] > audio.shape[2]:
        # swapping channel and time
        audio = audio.transpose(1, 2)
    if audio.shape[1] > 2:
        warnings.warn("Channel count > 2!. Only the first two channels " "will be processed!")
        audio = audio[..., :2]

    if audio.shape[1] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=1)

    if rate != model_rate:
        warnings.warn("resample to model sample rate")
        # we have to resample to model samplerate if needed
        # this makes sure we resample input only once
        resampler = torchaudio.transforms.Resample(
            orig_freq=rate, new_freq=model_rate, resampling_method="sinc_interpolation"
        ).to(audio.device)
        audio = resampler(audio)
    return audio
