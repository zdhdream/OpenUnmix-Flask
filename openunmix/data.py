import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable

import torch
import torch.utils.data
import torchaudio
import tqdm


def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    加载音频的元数据信息。该函数的参数是一个字符串类型的路径(`path`),表示音频文件的路径。
    函数的返回值是一个字典,包含音频文件的以下信息：`samplerate(采样率)`,`samples(样本数)`和`duration(持续时间,以杪为单位)`
    """
    # get length of file in samples
    # 首先,函数检查当前使用的音频后端是否为`sox`,如果是,则抛出一个错误,因为“sox”后端已经不再被支持
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    # 使用`torchaudio.info`函数读取音频文件的信息,并将返回的结果存储在变量`si`当中
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
        path: str,
        start: float = 0.0,
        dur: Optional[float] = None,
        info: Optional[dict] = None,
):
    """Load audio file

    Args:
        path: Path of audio file   音频文件的路径
        start: start position in seconds, defaults on the beginning. 开始的位置，以秒为单位，默认值为 0.0。
        dur: end position in seconds, defaults to `None` (full file). ：结束的位置，以秒为单位，默认值为 None（即完整的文件）。
        info: metadata object as called from `load_info`. 从 load_info 函数中获取的元数据对象。

    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    # 判断`dur` 是否为`None`来确定加载整个音频文件还是部分音频文件
    # 如果`dur`为`None`,加载整个音频文件,否则就加载一部分音频文件
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:  # 加载部分音频文件时,需要先从`info`中获取采样率
        if info is None:
            info = load_info(path)
        # 计算出需要加载的帧数和偏移帧数
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)
        return sig, rate


def aug_from_str(list_of_function_names: list):
    """
    Args:
        list_of_function_names: 一组音频增强函数的名字
    Returns:

    如果该列表不为空,那么函数将返回一个由该列表中的函数组成的复合函数,该复合函数用于对音频数据进行处理
    复合函数的构造方法是:通过列表推导式,把列表中的函数通过globals()["_augment_" + aug]的方式访问到，
    然后使用Compose类把它们组合在一起。
    如果该列表为空，那么函数将返回一个简单的函数，该函数直接返回其输入的音频数据。
    """
    if list_of_function_names:
        # 从全局命名空间(即Python程序的全局作用域)中取出名为`_augment_ + aug` 对象
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    该类用于组合多个音频增强变换(augmentation transforms)
    Args:
        augmentations: list of augmentations to compose.
        transforms: 包含多个变换函数的列表
    流程:
        (1): 初始化一个`Compose`实例,并传入变换函数列表
        (2): 调用该实例,将一个音频数据作为参数传入
        (3): 该实例逐个应用列表中的每个变换函数,最终返回处理后的音频数据
    """

    def __init__(self, transforms):
        self.transforms = transforms

    # 该类重写了 Python 的魔注册方法 __call__，
    # 可以把一个 Compose 类的实例当作一个函数来调用，
    # 这意味着该类实例可以像函数一样被调用。
    # 该方法接收一个参数 audio，并对该参数的音频数据逐个应用列表中的每个变换函数，
    # 最终返回处理后的音频数据。
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`
    作用: 对传入的`audio`参数进行数据增强处理。
        具体来说,它会生成一个随机数`g`,这个随机数在`low`和`high`之间。
        然后将`audio`和`g`这个随机数相乘,最终返回结果。
    可以看出，这个函数实现了音频数据的音量增强。其中`low`和`high`分别代表增强的下限和上限
    即音量的最小增强倍数和最大增强倍数。在函数内,通过`torch.rand(1)`生成一个在[0,1)范围内的随机数
    然后乘上(`high`-`low`)得到在(`low`,`high`)范围内的随机数
    """
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5
    作用：用于交换音频信号的左右声道
    该函数的输入是为一个形状为 (num_channels, num_samples) 的 PyTorch 张量，即音频波形张量。
    返回值是交换声道后的音频张量,形状与原始音频张量相同。
    """
    # 如果音频信号是立体声(即声道数为2)且随机生成的数在[0,0.5)之间,则对音频张量进行翻转,交换左右声道。否则,返回原始的音频张量
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        # torch.flip: 反转tensor的某些维度,反转tensor的第0维
        return torch.flip(audio, [0])
    else:
        return audio


def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    """
    作用:强制将音频变成立体声
    """
    # for multichannel > 2, we drop the other channels
    # 如果音频的通道数大于2,则丢弃多余的通道
    if audio.shape[0] > 2:
        audio = audio[:2, ...]
    # 如果音频的通道数为1,则复制它,使其变成立体声
    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio


class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: Union[Path, str],  # 数据集的跟目录
            sample_rate: float,  # 数据的采样频率
            seq_duration: Optional[float] = None,  # 数据的长度,单位是秒
            source_augmentations: Optional[Callable] = None,  # 用于数据增强的回调函数
    ) -> None:
        self.root = Path(args.root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        该方法在打印该类的实例时会被调用，用于生成一个可读的字符串，该字符串描述了该类的信息。
        """
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """
        用于生成存储额外信息的字符串，该信息将作为类的表示字符串的一部分。
        """
        return ""


def load_datasets(
        parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Tuple[UnmixDataset, UnmixDataset, argparse.Namespace]:
    """Loads the specified dataset from commandline arguments
    加载用户指定的数据集,并返回训练数据和验证数据集。它使用argparse模块读取命令行参数并决定如何创建不同类型的数据集
    代码中有五个不同类型的数据集: aligned, sourcefolder, tarckfolder_fix, trackfolder_var 和 reverb
    用户可以在命令行中通过 --dataset参数选择使用哪个数据集。对于每种数据集类型，代码都定义了一些额外的参数,并使用argparse创建相应的数据集
    `aligned`: 这个数据集包含被预先处理过的音频数据，他们已经通过一种方法被对齐到一起。这样做的目的是保证每一个音频文件的时长相同,以方便更好地训练模型
    音频文件的时间轴已经被调整，使得它们在时间上同步，如果两个音频文件都是从第一秒开始播放，那么对齐后两个音频文件的第一秒应该是完全相同的
    这种对齐的过程可以通过一些音频处理技术实现，例如通过音频相关性分析来确定音频文件的同步点，然后调整这些音频文件的时间轴使得它们在时间上同步
    `sourcefolder`: 这个数据集包含原始地音频文件,这些文件未经过任何处理
    `trackfolder_fix`: 这个数据集包含已经固定地音频数据，它们已经通过一些方法(例如通过降噪)处理过,但是仍然有一些固定不变地音频特征
    "已经固定的音频数据"：指的是音频中的音调、音量等元素已经同一处理过了，也就是说音频中的各种音色和音高已经被统一到了一定的范围内。这样处理过的音频数据可以更好地用于音频分离任务
    `trackfolder_var`: 这个数据集包含有变化的音频数据，它们可能包含一些不同的音频特征，如音频的音调、音高等。
    `reverb`: 这个数据集包含反射和混响效果的音频数据，它们是通过在音频中添加特定的效果而生成的。
    通过使用这五种不同类型的数据集，OpenUNMIX 模型可以学习到不同环境下音频的不同特征，
    从而提高对多种音频环境的适应能力，并且在将来的音频分离任务中表现更好
    Returns:
        train_dataset, validation_dataset
    """
    if args.dataset == "aligned":
        parser.add_argument(
            "--input-file",
            default="mixture.wav",
            type=str
        )
        parser.add_argument(
            "--output-file",
            default="other.wav",
            type=str
        )

        args = parser.parse_args()
        # set output target to basename of output file
        args.target = Path(args.output_file).stem

        dataset_kwargs = {
            "root": Path(args.root),
            "seq_duration": args.seq_dur,
            "input_file": args.input_file,
            "output_file": args.output_file,
        }
        # stem属性返回路径的主体部分，即文件名去除扩展名的部分
        args.target = Path(args.output_file).stem
        train_dataset = AlignedDataset(
            split="train", random_chunks=True, **dataset_kwargs
        )  # type: UnmixDataset
        valid_dataset = AlignedDataset(split="valid", **dataset_kwargs)  # type: UnmixDataset

    elif args.dataset == "sourcefolder":
        parser.add_argument("--interferer-dirs", type=str, nargs="+")
        parser.add_argument("--target-dir", type=str)
        parser.add_argument("--ext", type=str, default=".wav")
        parser.add_argument("--nb-train-samples", type=int, default=1000)
        parser.add_argument("--nb-valid-samples", type=int, default=100)
        parser.add_argument("--source-augmentations", type=str, nargs="+")
        args = parser.parse_args()
        args.target = args.target_dir

        dataset_kwargs = {
            "root": Path(args.root),
            "interferer_dirs": args.interferer_dirs,
            "target_dir": args.target_dir,
            "ext": args.ext,
        }

        source_augmentations = aug_from_str(args.source_augmentations)

        train_dataset = SourceFolderDataset(
            split="train",
            source_augmentations=source_augmentations,
            random_chunks=True,
            nb_samples=args.nb_train_samples,
            seq_duration=args.seq_dur,
            **dataset_kwargs,
        )

        valid_dataset = SourceFolderDataset(
            split="valid",
            random_chunks=True,
            seq_duration=args.seq_dur,
            nb_samples=args.nb_valid_samples,
            **dataset_kwargs,
        )

    elif args.dataset == "trackfolder_fix":
        parser.add_argument("--target-file", type=str)
        parser.add_argument("--interferer-files", type=str, nargs="+")
        parser.add_argument(
            "--random-track-mix",
            action="store_true",
            default=False,
            help="Apply random track mixing augmentation",
        )
        parser.add_argument("--source-augmentations", type=str, nargs="+")

        args = parser.parse_args()
        args.target = Path(args.target_file).stem

        dataset_kwargs = {
            "root": Path(args.root),
            "interferer_files": args.interferer_files,
            "target_file": args.target_file,
        }

        source_augmentations = aug_from_str(args.source_augmentations)

        train_dataset = FixedSourcesTrackFolderDataset(
            split="train",
            source_augmentations=source_augmentations,
            random_track_mix=args.random_track_mix,
            random_chunks=True,
            seq_duration=args.seq_dur,
            **dataset_kwargs,
        )
        valid_dataset = FixedSourcesTrackFolderDataset(
            split="valid", seq_duration=None, **dataset_kwargs
        )

    elif args.dataset == "trackfolder_var":
        parser.add_argument("--ext", type=str, default=".wav")
        parser.add_argument("--target-file", type=str)
        parser.add_argument("--source-augmentations", type=str, nargs="+")
        parser.add_argument(
            "--random-interferer-mix",
            action="store_true",
            default=False,
            help="Apply random interferer mixing augmentation",
        )
        parser.add_argument(
            "--silence-missing",
            action="store_true",
            default=False,
            help="silence missing targets",
        )

        args = parser.parse_args()
        args.target = Path(args.target_file).stem

        dataset_kwargs = {
            "root": Path(args.root),
            "target_file": args.target_file,
            "ext": args.ext,
            "silence_missing_targets": args.silence_missing,
        }

        source_augmentations = Compose(
            [globals()["_augment_" + aug] for aug in args.source_augmentations]
        )

        train_dataset = VariableSourcesTrackFolderDataset(
            split="train",
            source_augmentations=source_augmentations,
            random_interferer_mix=args.random_interferer_mix,
            random_chunks=True,
            seq_duration=args.seq_dur,
            **dataset_kwargs,
        )
        valid_dataset = VariableSourcesTrackFolderDataset(
            split="valid", seq_duration=None, **dataset_kwargs
        )

    else:
        parser.add_argument(
            "--is-wav",
            action="store_true",
            default=False,
            help="loads wav instead of STEMS",
        )
        parser.add_argument("--samples-per-track", type=int, default=64)
        parser.add_argument(
            "--source-augmentations", type=str, default=["gain", "channelswap"], nargs="+"
        )

        args = parser.parse_args()
        dataset_kwargs = {
            "root": args.root,
            "is_wav": args.is_wav,
            "subsets": "train",
            "target": args.target,
            "download": args.root is None,
            "seed": args.seed,
        }

        source_augmentations = aug_from_str(args.source_augmentations)

        train_dataset = MUSDBDataset(
            split="train",
            samples_per_track=args.samples_per_track,
            seq_duration=args.seq_dur,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            **dataset_kwargs,
        )

        valid_dataset = MUSDBDataset(
            split="valid", samples_per_track=1, seq_duration=None, **dataset_kwargs
        )

    return train_dataset, valid_dataset, args


class AlignedDataset(UnmixDataset):
    def __init__(
            self,
            root: str,  # 数据集路径
            split: str = "train",  # 数据集的类型
            input_file: str = "mixture.wav",  # 输入文件名
            output_file: str = "other.wav",  # 输出文件名
            seq_duration: Optional[float] = None,  # 每个样本音频片段的长度
            random_chunks: bool = False,  # 是否随机选择音频片段
            sample_rate: float = 44100.0,  # 采样率
            source_augmentations: Optional[Callable] = None,  # 对音频数据的预处理函数
            seed: int = 42,  # 随机数种子
    ) -> None:
        """A dataset of that assumes multiple track folders
        where each track includes and input and an output file
        which directly corresponds to the the input and the
        output of the model. This dataset is the most basic of
        all datasets provided here, due to the least amount of
        preprocessing, it is also the fastest option, however,
        it lacks any kind of source augmentations or custum mixing.

        Typical use cases:

        * Source Separation (Mixture -> Target)
        * Denoising (Noisy -> Clean)
        * Bandwidth Extension (Low Bandwidth -> High Bandwidth)

        Example
        =======
        data/train/01/mixture.wav --> input
        data/train/01/vocals.wav ---> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        # set the input and output files (accept glob)
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = list(self._get_paths())
        if not self.tuple_paths:
            raise RuntimeError("Dataset is empty, please check parameters")
        self.seed = seed
        random.seed(self.seed)

    def __getitem__(self, index):
        """
        可以获得数据集中索引为index的数据，
        其中如果设置了random_chunks为True，则随机选择序列，否则从头开始选择。
        最后，通过读取音频文件，获得输入和输出的音频数据，并将数据以张量的形式返回。
        """
        input_path, output_path = self.tuple_paths[index]

        if self.random_chunks:
            # 如果`random_chunks`为真,则在随机时间点从音频中截取长度为`seq_duration`的音频片段;否则,从音频的开头读取长度为`seq_duration`的音乐片段
            input_info = load_info(input_path)
            output_info = load_info(output_path)
            # duration是某个音频文件的播放时长，而self.seq_duration是指定的音频片段长度。通过随机生成一个浮点数来表示从音频文件的开头开始截取音频片段的起点。
            duration = min(input_info["duration"], output_info["duration"])
            start = random.uniform(0, duration - self.seq_duration)
        else:
            start = 0

        X_audio, _ = load_audio(input_path, start=start, dur=self.seq_duration)
        Y_audio, _ = load_audio(output_path, start=start, dur=self.seq_duration)
        # return torch tensors
        return X_audio, Y_audio

    def __len__(self):
        return len(self.tuple_paths)

    def _get_paths(self):
        """Loads input and output tracks
        获取输入和输出路径,其中路径在通过root和split目录下寻找包含输入和输出文件的文件夹得到的
        找到每个数据样本对应的输入音频文件和输出音频文件，并保存在列表`tuple_paths`中

        """
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                # 查找`track_path`路径下的所有与`self.input_file`模式匹配的文件
                input_path = list(track_path.glob(self.input_file))
                output_path = list(track_path.glob(self.output_file))
                if input_path and output_path:
                    if self.seq_duration is not None:  # 如果`seq_duration`不为空，代表需要对音频时长进行限制
                        # 因此需要读取输入音频文件和输出音频文件的信息，再比较它们的时长
                        input_info = load_info(input_path[0])
                        output_info = load_info(output_path[0])
                        min_duration = min(input_info["duration"], output_info["duration"])
                        # check if both targets are available in the subfolder
                        # 如果两个音频的最小时长都大于变量`seq_duration`，则表示该数据样本可以用于训练，则该样本的输入和输出路径作为一个元组返回
                        if min_duration > self.seq_duration:
                            yield input_path[0], output_path[0]
                    else:
                        yield input_path[0], output_path[0]


class SourceFolderDataset(UnmixDataset):
    def __init__(
            self,
            root: str,  # 存储音频数据的根目录
            split: str = "train",  # 数据集类型
            target_dir: str = "vocals",  # 存储目标音频数据的文件夹
            interferer_dirs: List[str] = ["bass", "drums"],  # 存储干扰音频数据的文件夹列表
            ext: str = ".wav",  # 音频文件扩展名
            nb_samples: int = 1000,  # 数据集中的样本数量
            seq_duration: Optional[float] = None,  # 生成样本的音频片段的长度
            random_chunks: bool = True,  # 指示是否从原始音频中随机选择片段
            sample_rate: float = 44100.0,  # 采样率
            source_augmentations: Optional[Callable] = lambda audio: audio,
            seed: int = 42,
    ) -> None:
        """A dataset that assumes folders of sources,
        instead of track folders. This is a common
        format for speech and environmental sound datasets
        such das DCASE. For each source a variable number of
        tracks/sounds is available, therefore the dataset
        is unaligned by design.
        By default, for each sample, sources from random track are drawn
        to assemble the mixture.

        Example
        =======
        train/vocals/track11.wav -----------------\
        train/drums/track202.wav  (interferer1) ---+--> input
        train/bass/track007a.wav  (interferer2) --/

        train/vocals/track11.wav ---------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.ext = ext
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.target_dir = target_dir
        self.interferer_dirs = interferer_dirs
        self.source_folders = self.interferer_dirs + [self.target_dir]
        self.source_tracks = self.get_tracks()
        self.nb_samples = nb_samples
        self.seed = seed
        random.seed(self.seed)

    def __getitem__(self, index):
        # For each source draw a random sound and mix them together
        """
        每个音频源从随机的音频轨道中选择一个，并将它们混合在一起。输出是混合音频和目标音频（最后一个音频源）
        """
        audio_sources = []
        for source in self.source_folders:
            if self.split == "valid":  # 如果当前处于验证集,则每次选择的音频轨道应该保证相同,以实现每个epoch选择相同的轨道的效果
                # provide deterministic behaviour for validation so that
                # each epoch, the same tracks are yielded
                random.seed(index)

            # select a random track for each source
            source_path = random.choice(self.source_tracks[source])
            duration = load_info(source_path)["duration"]
            # 对于每个音频源，如果random_chunks标志设置为True，则从随机的开始位置选择一个片段，否则使用中心片段。
            if self.random_chunks:
                # for each source, select a random chunk
                start = random.uniform(0, duration - self.seq_duration)
            else:
                # use center segment
                start = max(duration // 2 - self.seq_duration // 2, 0)

            audio, _ = load_audio(source_path, start=start, dur=self.seq_duration)
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)
        # 将所有音频源应用到一个Torch张量上
        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        # 线性混合所有的音频源
        x = stems.sum(0)
        # target is always the last element in the list
        y = stems[-1]
        return x, y

    def __len__(self):
        return self.nb_samples

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        source_tracks = {}
        for source_folder in tqdm.tqdm(self.source_folders):
            tracks = []
            source_path = p / source_folder
            # 每个音频文件必须以ext(.wav)结尾
            for source_track_path in sorted(source_path.glob("*" + self.ext)):
                if self.seq_duration is not None:
                    info = load_info(source_track_path)
                    # get minimum duration of track
                    # 如果self.seq_duration不为空，则检查音轨的时长是否大于该变量的值。如果是，则将该音轨文件的路径添加到音轨列表中。
                    if info["duration"] > self.seq_duration:
                        tracks.append(source_track_path)
                else:  # 如果self.seq_duration为空，则直接将该音轨文件的路径添加到音轨列表中。
                    tracks.append(source_track_path)
            # 最终，函数将音轨列表存储在以音频源名称为键的字典中，并返回该字典。
            source_tracks[source_folder] = tracks
        return source_tracks


class FixedSourcesTrackFolderDataset(UnmixDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_file: str = "vocals.wav",
            interferer_files: List[str] = ["bass.wav", "drums.wav"],
            seq_duration: Optional[float] = None,
            random_chunks: bool = False,
            random_track_mix: bool = False,
            source_augmentations: Optional[Callable] = lambda audio: audio,
            sample_rate: float = 44100.0,
            seed: int = 42,
    ) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.

        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.

        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.

        Example
        =======
        train/1/vocals.wav ---------------\
        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/

        train/1/vocals.wav -------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = self.interferer_files + [self.target_file]
        self.seed = seed
        random.seed(self.seed)

        self.tracks = list(self.get_tracks())
        if not len(self.tracks):
            raise RuntimeError("No tracks found")

    def __getitem__(self, index):
        # first, get target track
        track_path = self.tracks[index]["path"]
        min_duration = self.tracks[index]["min_duration"]
        if self.random_chunks:
            # determine start seek by target duration
            start = random.uniform(0, min_duration - self.seq_duration)
        else:
            start = 0

        # assemble the mixture of target and interferers
        audio_sources = []
        # load target
        target_audio, _ = load_audio(
            track_path / self.target_file, start=start, dur=self.seq_duration
        )
        target_audio = self.source_augmentations(target_audio)
        audio_sources.append(target_audio)
        # load interferers
        for source in self.interferer_files:
            # optionally select a random track for each source
            if self.random_track_mix:
                random_idx = random.choice(range(len(self.tracks)))
                track_path = self.tracks[random_idx]["path"]
                if self.random_chunks:
                    min_duration = self.tracks[random_idx]["min_duration"]
                    start = random.uniform(0, min_duration - self.seq_duration)

            audio, _ = load_audio(track_path / source, start=start, dur=self.seq_duration)
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the first element in the list
        y = stems[0]
        return x, y

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track ", track_path)
                    continue

                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    # get minimum duration of track
                    min_duration = min(i["duration"] for i in infos)
                    if min_duration > self.seq_duration:
                        yield ({"path": track_path, "min_duration": min_duration})
                else:
                    yield ({"path": track_path, "min_duration": None})


class VariableSourcesTrackFolderDataset(UnmixDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_file: str = "vocals.wav",
            ext: str = ".wav",
            seq_duration: Optional[float] = None,
            random_chunks: bool = False,
            random_interferer_mix: bool = False,
            sample_rate: float = 44100.0,
            source_augmentations: Optional[Callable] = lambda audio: audio,
            silence_missing_targets: bool = False,
    ) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a _variable_ number of sources.
        The users specifies the target file-name (`target_file`)
        and the extension of sources to used for mixing.
        A linear mix is performed on the fly by summing all sources in a
        track folder.

        Since the number of sources differ per track,
        while target is fixed, a random track mix
        augmentation cannot be used. Instead, a random track
        can be used to load the interfering sources.

        Also make sure, that you do not provide the mixture
        file among the sources!

        Example
        =======
        train/1/vocals.wav --> input target   \
        train/1/drums.wav --> input target     |
        train/1/bass.wav --> input target    --+--> input
        train/1/accordion.wav --> input target |
        train/1/marimba.wav --> input target  /

        train/1/vocals.wav -----------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        self.random_interferer_mix = random_interferer_mix
        self.source_augmentations = source_augmentations
        self.target_file = target_file
        self.ext = ext
        self.silence_missing_targets = silence_missing_targets
        self.tracks = list(self.get_tracks())

    def __getitem__(self, index):
        # select the target based on the dataset   index
        target_track_path = self.tracks[index]["path"]
        if self.random_chunks:
            target_min_duration = self.tracks[index]["min_duration"]
            target_start = random.uniform(0, target_min_duration - self.seq_duration)
        else:
            target_start = 0

        # optionally select a random interferer track
        if self.random_interferer_mix:
            random_idx = random.choice(range(len(self.tracks)))
            intfr_track_path = self.tracks[random_idx]["path"]
            if self.random_chunks:
                intfr_min_duration = self.tracks[random_idx]["min_duration"]
                intfr_start = random.uniform(0, intfr_min_duration - self.seq_duration)
            else:
                intfr_start = 0
        else:
            intfr_track_path = target_track_path
            intfr_start = target_start

        # get sources from interferer track
        sources = sorted(list(intfr_track_path.glob("*" + self.ext)))

        # load sources
        x = 0
        for source_path in sources:
            # skip target file and load it later
            if source_path == intfr_track_path / self.target_file:
                continue

            try:
                audio, _ = load_audio(source_path, start=intfr_start, dur=self.seq_duration)
            except RuntimeError:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            x += self.source_augmentations(audio)

        # load the selected track target
        if Path(target_track_path / self.target_file).exists():
            y, _ = load_audio(
                target_track_path / self.target_file,
                start=target_start,
                dur=self.seq_duration,
            )
            y = self.source_augmentations(y)
            x += y

        # Use silence if target does not exist
        else:
            y = torch.zeros(audio.shape)

        return x, y

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                # check if target exists
                if Path(track_path, self.target_file).exists() or self.silence_missing_targets:
                    sources = sorted(list(track_path.glob("*" + self.ext)))
                    if not sources:
                        # in case of empty folder
                        print("empty track: ", track_path)
                        continue
                    if self.seq_duration is not None:
                        # check sources
                        infos = list(map(load_info, sources))
                        # get minimum duration of source
                        min_duration = min(i["duration"] for i in infos)
                        if min_duration > self.seq_duration:
                            yield ({"path": track_path, "min_duration": min_duration})
                    else:
                        yield ({"path": track_path, "min_duration": None})


class MUSDBDataset(UnmixDataset):
    def __init__(
            self,
            target: str = "vocals",
            root: str = None,
            download: bool = False,
            is_wav: bool = False,
            subsets: str = "train",
            split: str = "train",
            seq_duration: Optional[float] = 6.0,
            samples_per_track: int = 64,
            source_augmentations: Optional[Callable] = lambda audio: audio,
            random_track_mix: bool = False,
            seed: int = 42,
            *args,
            **kwargs,
    ) -> None:
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.
        MUSDB18 torch.data.Dataset 从 MUSDB tracks 采样 使用轨道和摘录替换。

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
            要分离的源的目标名称，默认为 ``vocals``。
        root : str
            root path of MUSDB
            musdb根路径。如果设置为None，它将从MUSDB_PATH环境变量中读取
        download : boolean
            automatically download 7s preview version of MUSDB
            下载MUSDB18的样本版本，其中包括7s摘录，默认为False
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
            期望每个源都有wav文件的子文件夹，而不是轨道，默认为False
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
            选择_musdb_子集训练或测试。默使用训练集
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
            训练以 ``seq_duration`` 的块进行（以秒为单位，默认为 ``None`` 加载完整的音轨
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
            设置样本数，每个时期从每个轨道产生。默认为 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
            提供将形状为 (src, samples) 的多通道音频文件作为输入和输出的增强函数列表。 默认为无增强（输入=输出）
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
            随机混合来自不同轨道的源来组装一个自定义组合。 此增强仅适用于 train subset。
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.
            用于为 musdb 数据集初始化函数添加进一步控制的附加关键字参数。

        """
        import musdb

        self.seed = seed
        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args,
            **kwargs,
        )
        self.sample_rate = 44100.0  # musdb is fixed sample rate

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == "train" and self.seq_duration:
            for k, source in enumerate(self.mus.setup["sources"]):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration

                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
                # load source audio and apply time domain source_augmentations
                audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup["sources"].keys()).index("vocals")
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )

    parser.add_argument("--root", type=str, help="root path of dataset")

    parser.add_argument(
        "--save", action="store_true", help=("write out a fixed dataset of samples")
    )

    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # I/O Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=5.0,
        help="Duration of <=0.0 will result in the full audio",
    )

    parser.add_argument("--batch-size", type=int, default=16)

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)

    train_dataset, valid_dataset, args = load_datasets(parser, args)
    print("Audio Backend: ", torchaudio.get_audio_backend())

    # Iterate over training dataset and compute statistics
    total_training_duration = 0
    for k in tqdm.tqdm(range(len(train_dataset))):
        x, y = train_dataset[k]
        total_training_duration += x.shape[1] / train_dataset.sample_rate
        if args.save:
            torchaudio.save("test/" + str(k) + "x.wav", x.T, train_dataset.sample_rate)
            torchaudio.save("test/" + str(k) + "y.wav", y.T, train_dataset.sample_rate)

    print("Total training duration (h): ", total_training_duration / 3600)
    print("Number of train samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = args.seq_dur

    train_sampler = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    for x, y in tqdm.tqdm(train_sampler):
        print(x.shape)
