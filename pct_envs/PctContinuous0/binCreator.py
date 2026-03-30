import numpy as np
import copy
import torch

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)

class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])

def _normalize_box_trajs(raw):
    """兼容两种格式：① 多条轨迹 ② 扁平单条轨迹 [box1,box2,...]（每条 box 为 [x,y,z] 或 [x,y,z,d]）"""
    if not raw:
        return []
    first = raw[0]
    try:
        if len(first) in (3, 4) and all(isinstance(x, (int, float)) for x in first):
            return [raw]
    except (TypeError, IndexError):
        pass
    return raw


class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        print("load data set successfully!")
        self.index = 0
        self.box_index = 0
        raw = torch.load(self.data_name)
        self.box_trajs = _normalize_box_trajs(raw)
        self.traj_nums = len(self.box_trajs)

    def reset(self, index=None):
        self.box_list.clear()
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        # 循环使用轨迹，防止索引越界
        self.index = self.index % self.traj_nums
        traj = self.box_trajs[self.index]
        self.boxes = np.array(traj)
        self.boxes = self.boxes.tolist()
        if self.boxes and not isinstance(self.boxes[0], (list, tuple)):
            self.boxes = []
        elif self.boxes and len(self.boxes[0]) not in (3, 4):
            self.boxes = [list(b)[:4] for b in self.boxes if hasattr(b, '__len__') and len(b) >= 3]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([100, 100, 100, 1.0])  # 添加 density 以支持 setting3

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10, 1.0))  # 添加 density 以支持 setting3
            self.recorder.append((10, 10, 10, 1.0))
            self.box_index += 1
