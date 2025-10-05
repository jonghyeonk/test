import sys
import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset


def trim_eog_data(eog_data, trim_length=100):
    heog_data = eog_data[:, 0]  # HEOG (750,)
    veog_data = eog_data[:, 1]  # VEOG (750,)

    trimmed_heog = heog_data[trim_length:-trim_length]  # (750 - 2 * trim_length,)
    trimmed_veog = veog_data[trim_length:-trim_length]  # (750 - 2 * trim_length,)

    trimmed_eog_data = torch.stack((trimmed_heog, trimmed_veog), dim=-1)  # (seq_len - 2 * trim_length, 2)

    return trimmed_eog_data


def sliding_multi(x, size=30, wait=150, alpha=0.5):
    gap_hist = []
    save = []
    for i in range(wait, len(x) - size + 1 - 10):
        gap = 0
        window = x[i:(i + size - 1)]
        seemax = np.max(window)
        seemin = np.min(window)
        see = seemax - seemin
        save.append(see)

        if i > wait + size:
            cond = ((see > save[len(save) - 2] + alpha * np.std(save[0:(len(save) - 1)])) and (
                    np.std(save[0:(len(save) - 1)]) > 0))
            if cond:
                # amax = x[(i):(i + size - 1)].index(seemax)
                # amin = x[(i):(i + size - 1)].index(seemin)

                amax = np.argmax(window)  # 최대값 인덱스
                amin = np.argmin(window)  # 최소값 인덱스

                if amax < amin:
                    gap = -1 * see
                else:
                    gap = see

                for n in range(10):
                    window = x[(i + n + 1):(i + n + size)]
                    seemax = np.max(window)
                    seemin = np.min(window)
                    amax = np.argmax(window)
                    amin = np.argmin(window)
                    see = seemax - seemin
                    if amax < amin:
                        gap = -1 * see
                    else:
                        gap = see
        gap_hist.append(gap)

    return gap_hist


class EOGPreprocessDataset(Dataset):
    def __init__(self, root='./EOG-Data', split='test', padding_length=751, augmentation=False, args=None):
        if split is not None:
            self.root = os.path.join(root, split)
        else:
            self.root = root

        self.csv_list = os.listdir(self.root)
        self.csv_files = [file for file in self.csv_list if file.endswith(".csv")]
        # print(self.csv_files)
        self.csv_file_paths = [os.path.join(self.root, csv_file) for csv_file in self.csv_files]
        self.padding_length = padding_length
        self.args = args
        print(f"{split} counts:{len(self.csv_file_paths)}")

    def __getitem__(self, index):
        if index < 0 or index >= len(self.csv_file_paths):
            return None
        csv_path = self.csv_file_paths[index]

        try:
            data = pd.read_csv(csv_path, delimiter=',')
            # print(data)
            if self.args.dataset_name == 'task2_v2_x8_2' or self.args.dataset_name == 'task2_v2_x8_iot':
                heog = data['Dev1/ai3'].values
                veog = data['Dev1/ai0'].values
            else:
                # 1열이 h, 2열이 v
                heog = data['Dev1.ai3'].values
                veog = data['Dev1.ai0'].values

            org_heog = heog.copy()
            org_veog = veog.copy()

            eog_data = np.vstack((heog, veog)).T

            data = torch.tensor(eog_data, dtype=torch.float32)
            data = trim_eog_data(data, trim_length=50)
            trim_data_np = data.numpy()

            multigap_H = sliding_multi(trim_data_np[:, 0], 30, 0, 0.5)
            multigap_V = sliding_multi(trim_data_np[:, 1], 30, 0, 0.5)

            multigap_H_tensor = torch.tensor(multigap_H, dtype=torch.float32)
            multigap_V_tensor = torch.tensor(multigap_V, dtype=torch.float32)

            diff_data = torch.stack([multigap_H_tensor, multigap_V_tensor], dim=-1)
            data_trimmed_for_stack = data[:diff_data.shape[0], :]

            data = torch.stack(
                [data_trimmed_for_stack[:, 0], data_trimmed_for_stack[:, 1], diff_data[:, 0], diff_data[:, 1]], dim=-1
            )

            if data.size(0) < self.padding_length:
                padding = torch.zeros((self.padding_length - data.size(0), data.size(1)))
                data = torch.cat((data, padding), dim=0)
            elif data.size(0) > self.padding_length:
                data = data[:self.padding_length, :]

            # 데이터가 유효하지 않을 경우 처리
            if data is None:
                raise ValueError(f"Invalid data at index {index}")

            if sys.platform == 'win32' or sys.platform == 'win64':
                file_name = csv_path.split('\\')[-1]
            else:
                file_name = csv_path.split('/')[-1]

            if self.args.dataset_name == 'task2_v2_x8_iot':
                label_str = file_name.split('_')[2]
                start_x = 0.0
                start_y = 0.0
            elif 'iot' in self.root or "IOT" in self.root:
                label_str = file_name.split('_')[2]
                start_x = 0.0
                start_y = 0.0
            elif self.args.dataset_name == 'eskin':
                label_str = file_name.split('_')[2]
                start_x = 0.0
                start_y = 0.0
            else:
                label_str = file_name.split('_')[-1].replace('.csv', '')
                start_x = float(file_name.split('_')[1])
                start_y = float(file_name.split('_')[2])

            label = int(label_str)
            label = label - 1

            label_tensor = torch.tensor(label, dtype=torch.long)
            start_x = torch.tensor([start_x], dtype=torch.float32)
            start_y = torch.tensor([start_y], dtype=torch.float32)
            start = torch.tensor([[start_x, start_y]], dtype=torch.float32)

            return start, data, label_tensor, csv_path, (org_heog, org_veog)
        except Exception as e:
            print(f"Error at {csv_path}: {e}")
            return None

    def __len__(self):
        return len(self.csv_list)

    def collate_fn(self, batch):
        # original_len = len(batch)
        batch = [item for item in batch if item is not None]

        if not batch or len(batch) == 0:
            pass
        else:
            start, data, label, csv_path, org_data = zip(*batch)
            start = torch.stack(start, dim=0)
            label = torch.stack(label, dim=0)
            padded_data = torch.zeros(len(data), self.padding_length, data[0].shape[1])  # 데이터의 두 번째 차원 크기에 주의

            for i, d in enumerate(data):
                padded_data[i, :min(d.shape[0], self.padding_length), :] = d[:self.padding_length, :]

            return start, padded_data, label, csv_path, org_data
