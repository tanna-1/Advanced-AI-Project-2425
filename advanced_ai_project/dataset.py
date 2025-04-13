from torch.utils.data import Dataset
import mmap


class CharIndexDataset(Dataset):
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

    def __getitem__(self, idx: int):
        return idx, self.__mapped_file[idx]

    def __del__(self):
        if hasattr(self, "mapped_file"):
            self.__mapped_file.close()


class PromptDataset(Dataset):
    def __init__(self, prompt: str, index_offset: int = 0):
        self.__index_offset = index_offset
        self.__prompt = prompt.encode("utf-8")

    def __len__(self) -> int:
        return len(self.__prompt)

    def __getitem__(self, idx: int):
        return self.__index_offset + idx, self.__prompt[idx]
