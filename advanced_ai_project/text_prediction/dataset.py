import mmap
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


# TokenDataset yields a token and its corresponding index
class TokenDataset(Dataset[tuple[int, int]], ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[int, int]: ...


class ByteFileDataset(TokenDataset):
    def __init__(self, file_path: str, length_cutoff: int | None = None):
        with open(file_path, "rb") as f:
            self.__mapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.__file_length = len(self.__mapped_file)

        if length_cutoff is not None:
            if length_cutoff > self.__file_length:
                raise ValueError(
                    f"length_cutoff {length_cutoff} is greater than file length {self.__file_length}"
                )
            self.__file_length = length_cutoff

    def __len__(self) -> int:
        return self.__file_length

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return idx, self.__mapped_file[idx]

    def __del__(self):
        if hasattr(self, "mapped_file"):
            self.__mapped_file.close()


class StringDataset(TokenDataset):
    def __init__(self, prompt: str, start_index: int = 0):
        self.__start_index = start_index
        self.__prompt = prompt.encode("utf-8")

    def __len__(self) -> int:
        return len(self.__prompt)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.__start_index + idx, self.__prompt[idx]


class SingleTokenDataset(TokenDataset):
    def __init__(self, token: int, start_index: int = 0):
        self.__start_index = start_index
        self.__token = token

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.__start_index, self.__token
