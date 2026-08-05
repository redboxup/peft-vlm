import torch
from typing import Tuple, List, Any

from utils.reproducibility import deterministic_randperm


class GlobalBatchShuffler:
    @staticmethod
    def _get_batchsizes(
        total_length: int, batch_size: int
    ) -> List[int]:
        full_batches = total_length // batch_size
        remainder = total_length % batch_size
        sizes = [batch_size] * full_batches
        if remainder:
            sizes.append(remainder)
        return sizes

    @staticmethod
    def _flatten_batches(batches: List[Any]) -> Tuple[Any, int]:
        if isinstance(batches[0], torch.Tensor):
            flat_data = torch.cat(batches, dim=0)
            return flat_data, flat_data.size(0)

        elif isinstance(batches[0], (tuple, list)):
            flat_data = tuple(
                torch.cat(parts, dim=0) for parts in zip(*batches)
            )
            return flat_data, flat_data[0].size(0)

        elif isinstance(batches[0], dict):
            keys = batches[0].keys()
            flat_data = {
                key: torch.cat([batch[key] for batch in batches], dim=0)
                for key in keys
            }
            return flat_data, next(iter(flat_data.values())).size(0)

        else:
            raise TypeError(f"Unsupported data format: {type(batches[0])}")

    @staticmethod
    def _shuffle(flat_data: Any, total: int, seed: int) -> Any:
        perm = deterministic_randperm(length=total, seed=seed)

        if isinstance(flat_data, torch.Tensor):
            return flat_data[perm]

        elif isinstance(flat_data, tuple):
            return tuple(part[perm] for part in flat_data)

        elif isinstance(flat_data, dict):
            return {key: value[perm] for key, value in flat_data.items()}

    @staticmethod
    def _create_batches(flat_data: Any, batch_sizes: List[int]) -> List[Any]:
        indices = [0] + list(
            torch.cumsum(torch.tensor(batch_sizes), dim=0).tolist()
        )

        if isinstance(flat_data, torch.Tensor):
            return [
                flat_data[indices[i]: indices[i+1]]
                for i in range(len(batch_sizes))
            ]

        elif isinstance(flat_data, tuple):
            return [
                tuple(part[indices[i]:indices[i+1]] for part in flat_data)
                for i in range(len(batch_sizes))
            ]

        elif isinstance(flat_data, dict):
            return [
                {
                    key: flat_data[key][indices[i]: indices[i+1]]
                    for key in flat_data
                }
                for i in range(len(batch_sizes))
            ]

        else:
            raise TypeError(f"Unsupported data type: {type(flat_data)}")

    @staticmethod
    def shuffle_batches(
        list_batches: List[Any],
        batch_size: int,
        total_len: int,
        seed: int = 42,
    ) -> Any:
        # if list is empty
        if not list_batches:
            return []

        batch_sizes = (
            GlobalBatchShuffler._get_batchsizes(
                total_length=total_len, batch_size=batch_size,
            )
        )

        flat_data, total = GlobalBatchShuffler._flatten_batches(list_batches)

        flat_data = GlobalBatchShuffler._shuffle(
            flat_data, total, seed=seed,
        )

        new_batches = GlobalBatchShuffler._create_batches(
            flat_data=flat_data, batch_sizes=batch_sizes,
        )

        return new_batches
