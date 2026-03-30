# -*- coding: utf-8 -*-
"""
杞ㄨ抗寮忔粦鍔ㄧ獥鍙ｇ幆澧冨寘瑁呭櫒
浠庡垎缁勮建杩规暟鎹腑璇诲彇锛屾瘡涓?episode 瀵瑰簲涓€鏉¤建杩癸紝缁存姢5涓寘瑁圭殑鍊欓€夌獥鍙?
涓庡師濮?PCT 鐨?LoadBoxCreator 浣跨敤鏂瑰紡涓€鑷?

鏁版嵁鏍煎紡鍏煎锛?
  - 鍒嗙粍杞ㄨ抗: [[traj1], [traj2], ...]锛屾瘡鏉¤建杩规槸 [[w,h,d], [w,h,d], ...] 鍒楄〃
  - 鎵佸钩鍒楄〃: [[w,h,d], [w,h,d], ...]锛岃嚜鍔ㄨ涓轰竴鏉″ぇ杞ㄨ抗
"""
import numpy as np
import torch
import gym
from collections import deque


def normalize_to_trajectories(raw):
    """
    鍏煎涓ょ鏁版嵁鏍煎紡锛岀粺涓€杞负杞ㄨ抗鍒楄〃

    Args:
        raw: 鍘熷鏁版嵁锛屽彲浠ユ槸鍒嗙粍杞ㄨ抗鎴栨墎骞冲垪琛?

    Returns:
        list of trajectories, 姣忔潯杞ㄨ抗鏄?[[w,h,d], ...] 鍒楄〃
    """
    if not raw:
        return []
    first = raw[0]
    # 鑻ラ鏉℃槸闀垮害涓?3 鐨勬暟鍒楋紙涓€涓瀛愶級锛屽垯涓烘墎骞冲垪琛紝鏁翠唤褰撲綔涓€鏉¤建杩?
    try:
        if len(first) == 3 and isinstance(first[0], (int, float)):
            return [raw]
    except (TypeError, IndexError):
        pass
    return raw


class SlidingWindowEnvWrapper:
    """
    杞ㄨ抗寮忔粦鍔ㄧ獥鍙ｇ幆澧冨寘瑁呭櫒

    鏍稿績閫昏緫锛堜笌鍘熷 PCT LoadBoxCreator 瀵归綈锛?
    1. 鏁版嵁鎸夎建杩瑰垎缁勶紝姣忎釜 episode 瀵瑰簲涓€鏉¤建杩?
    2. 浠庡綋鍓嶈建杩逛腑鎸夐『搴忓彇鍑虹墿鍝侊紝濉厖5涓€欓€夌獥鍙?
    3. Set Transformer 浠?涓€欓€変腑閫?涓?
    4. PCT 鍐冲畾鏀惧湪鍝噷
    5. 鏀剧疆鎴愬姛 鈫?绉婚櫎璇ュ寘瑁癸紝浠庡綋鍓嶈建杩逛腑琛ュ厖鏂板寘瑁?
    6. 鏀剧疆澶辫触 鎴?杞ㄨ抗涓棤鏂扮墿鍝佸彲琛ュ厖涓旂獥鍙ｄ负绌?鈫?episode 缁撴潫
    7. 涓嬩竴涓?episode 浣跨敤涓嬩竴鏉¤建杩癸紙寰幆锛?
    """

    def __init__(self,
                 base_env,
                 trajectories,
                 window_size=5,
                 normFactor=0.0125,
                 traj_start_idx=0,
                 flatness_reward_coef=0.2,
                 flatness_interval=10,
                 flatness_min_placed=10,
                 flatness_grid_size=16):
        """
        Args:
            base_env: 鍘熷 PCT 鐜
            trajectories: 杞ㄨ抗鍒楄〃 [[traj1], [traj2], ...] (宸茬粡杩?normalize_to_trajectories)
            window_size: 婊戝姩绐楀彛澶у皬
            normFactor: 褰掍竴鍖栧洜瀛?(1/max(container_size))
            traj_start_idx: 鍒濆杞ㄨ抗绱㈠紩 (鐢ㄤ簬澶氳繘绋嬪悇鑷粠涓嶅悓杞ㄨ抗寮€濮?
        """
        self.base_env = base_env
        self.trajectories = trajectories
        self.num_trajectories = len(trajectories)
        self.window_size = window_size
        self.normFactor = normFactor
        self.flatness_reward_coef = float(flatness_reward_coef)
        self.flatness_interval = max(1, int(flatness_interval))
        self.flatness_min_placed = max(1, int(flatness_min_placed))
        self.flatness_grid_size = max(4, int(flatness_grid_size))

        # 瀹瑰櫒淇℃伅
        self.container_size = np.array(base_env.bin_size)

        # 婊戝姩绐楀彛闃熷垪
        self.candidate_queue = deque(maxlen=window_size)

        # 杞ㄨ抗绱㈠紩 鈥?褰撳墠浣跨敤鍝潯杞ㄨ抗
        self.traj_idx = traj_start_idx % self.num_trajectories

        # 杞ㄨ抗鍐呮寚閽?鈥?鎸囧悜褰撳墠杞ㄨ抗涓笅涓€涓鍙栧嚭鐨勭墿鍝?
        self.item_pointer = 0
        self.current_traj = []

        # 鐘舵€佽拷韪?
        self.total_placed = 0
        self.done = False
        self.last_selected_idx = 0

    def _compute_height_map(self):
        """
        Build a coarse top-surface height map from placed boxes.
        """
        g = self.flatness_grid_size
        heights = np.zeros((g, g), dtype=np.float32)
        boxes = getattr(self.base_env.space, 'boxes', [])
        if len(boxes) == 0:
            return heights

        width, length, _ = self.container_size
        cell_w = width / g
        cell_l = length / g

        for box in boxes:
            x0, y0 = float(box.lx), float(box.ly)
            x1, y1 = x0 + float(box.x), y0 + float(box.y)
            top_h = float(box.lz + box.z)

            ix0 = max(0, int(np.floor(x0 / cell_w)))
            ix1 = min(g - 1, int(np.ceil(x1 / cell_w) - 1))
            iy0 = max(0, int(np.floor(y0 / cell_l)))
            iy1 = min(g - 1, int(np.ceil(y1 / cell_l) - 1))
            if ix1 < ix0 or iy1 < iy0:
                continue

            for ix in range(ix0, ix1 + 1):
                cx0, cx1 = ix * cell_w, (ix + 1) * cell_w
                overlap_x = min(x1, cx1) - max(x0, cx0)
                if overlap_x <= 1e-9:
                    continue
                for iy in range(iy0, iy1 + 1):
                    cy0, cy1 = iy * cell_l, (iy + 1) * cell_l
                    overlap_y = min(y1, cy1) - max(y0, cy0)
                    if overlap_y <= 1e-9:
                        continue
                    if top_h > heights[ix, iy]:
                        heights[ix, iy] = top_h
        return heights

    def _compute_flatness_score(self):
        """
        Compute a long-horizon pallet flatness score in [0, 1].
        """
        height_map = self._compute_height_map()
        occupied = height_map > 0
        occupied_count = int(np.sum(occupied))
        if occupied_count == 0:
            return 0.0

        occ_heights = height_map[occupied]
        height_std = float(np.std(occ_heights))
        flatness_raw = 1.0 - height_std / (float(self.container_size[2]) + 1e-8)
        flatness_raw = float(np.clip(flatness_raw, 0.0, 1.0))

        coverage_ratio = float(occupied_count) / float(height_map.size)
        score = flatness_raw * (0.5 + 0.5 * coverage_ratio)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_long_term_flatness_reward(self):
        """
        Trigger flatness reward every `flatness_interval` placements.
        """
        if self.flatness_reward_coef <= 0:
            return 0.0, 0.0
        if self.total_placed < self.flatness_min_placed:
            return 0.0, 0.0
        if self.total_placed % self.flatness_interval != 0:
            return 0.0, 0.0

        flatness_score = self._compute_flatness_score()
        bonus = self.flatness_reward_coef * flatness_score
        return float(bonus), float(flatness_score)

    def _next_item_from_traj(self):
        """
        浠庡綋鍓嶈建杩逛腑鍙栧嚭涓嬩竴涓墿鍝?

        Returns:
            item: [w, h, d] 鍒楄〃锛屽鏋滆建杩瑰凡鐢ㄥ畬鍒欒繑鍥?None
        """
        if self.item_pointer >= len(self.current_traj):
            return None
        item = self.current_traj[self.item_pointer]
        self.item_pointer += 1
        return list(item)

    def _fill_window(self):
        """
        灏濊瘯浠庡綋鍓嶈建杩逛腑鍙栫墿鍝佸～婊″€欓€夌獥鍙?

        Returns:
            bool: 绐楀彛涓槸鍚︽湁鑷冲皯1涓墿鍝?
        """
        while len(self.candidate_queue) < self.window_size:
            item = self._next_item_from_traj()
            if item is None:
                break
            self.candidate_queue.append(item)
        return len(self.candidate_queue) > 0

    def reset(self, episode_idx=None):
        """
        閲嶇疆鐜 鈥?寮€濮嬫柊鐨勭瀛愬拰鏂扮殑杞ㄨ抗

        涓?LoadBoxCreator.reset() 绫讳技锛氭瘡娆?reset 鍙栦笅涓€鏉¤建杩?
        """
        self.total_placed = 0
        self.done = False
        self.last_selected_idx = 0

        # 閫夋嫨杞ㄨ抗
        self.current_traj = list(self.trajectories[self.traj_idx])
        self.item_pointer = 0

        # 杞浆鍒颁笅涓€鏉¤建杩?(涓嬫 reset 鏃朵娇鐢?
        self.traj_idx = (self.traj_idx + 1) % self.num_trajectories

        # 浠庤建杩逛腑鍙栫墿鍝佸～鍏呭€欓€夌獥鍙?
        self.candidate_queue.clear()
        self._fill_window()

        # 閲嶇疆搴曞眰 PCT 鐜 (娓呯┖绠卞瓙)
        self._set_next_box_in_env(0)
        obs = self.base_env.reset()

        return obs

    def _set_next_box_in_env(self, selected_idx):
        """
        灏嗛€変腑鐨勭瀛愯缃埌 PCT 鐜涓?
        """
        if len(self.candidate_queue) == 0:
            return None

        selected_idx = int(selected_idx)
        selected_idx = max(0, min(selected_idx, len(self.candidate_queue) - 1))

        selected_box = list(self.candidate_queue)[selected_idx]
        self.base_env.next_box = [round(selected_box[0], 3),
                                  round(selected_box[1], 3),
                                  round(selected_box[2], 3)]

        next_box = sorted(list(self.base_env.next_box))
        self.base_env.next_box_vec[:, 3:6] = next_box
        self.base_env.next_box_vec[:, 0] = 1  # density for setting 1
        self.base_env.next_box_vec[:, -1] = 1
        return selected_idx

    def set_selected_and_get_obs(self, selected_idx):
        """
        璁剧疆閫変腑鐨勫€欓€夊寘瑁癸紝骞堕噸鏂扮敓鎴?observation

        PCT 鐨?leaf_nodes 鏄熀浜?self.next_box 鐢熸垚鐨勶紝
        鎵€浠ュ繀椤诲湪璁剧疆 next_box 鍚庨噸鏂扮敓鎴?observation 鏉ヨ幏鍙栨纭殑 leaf_nodes
        """
        effective_idx = self._set_next_box_in_env(selected_idx)
        if effective_idx is not None:
            self.last_selected_idx = effective_idx
        obs = self._generate_observation_without_gen_next_box()
        return obs

    def _generate_observation_without_gen_next_box(self):
        """
        鐢熸垚 observation锛屼絾涓嶈皟鐢?gen_next_box()
        """
        boxes = []
        leaf_nodes = []

        saved_next_box = self.base_env.next_box
        saved_next_den = self.base_env.next_den

        boxes.append(self.base_env.space.box_vec)
        leaf_nodes.append(self.base_env.get_possible_position())

        next_box = sorted(list(saved_next_box))
        self.base_env.next_box_vec[:, 3:6] = next_box
        self.base_env.next_box_vec[:, 0] = saved_next_den
        self.base_env.next_box_vec[:, -1] = 1

        self.base_env.next_box = saved_next_box
        self.base_env.next_den = saved_next_den

        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.base_env.next_box_vec)), (-1))

    def get_candidates(self):
        """
        鑾峰彇褰撳墠鍊欓€夊寘瑁癸紙鍘熷灏哄锛屼笉鍋氬綊涓€鍖栵級

        Selector 闇€瑕佺湅鍒板師濮嬪昂瀵告墠鑳藉尯鍒嗕笉鍚屽€欓€?
        normFactor 鏄负 PCT 鐨勮娴嬬┖闂磋璁＄殑锛屼笉閫傜敤浜?selector

        Returns:
            candidates: [window_size, 3] numpy array (鍘熷灏哄)
        """
        candidates = np.zeros((self.window_size, 3))
        queue_list = list(self.candidate_queue)
        for i, box in enumerate(queue_list):
            candidates[i] = list(box)[:3]
        return candidates

    def get_container_state(self):
        """
        鑾峰彇瀹瑰櫒鐘舵€佺壒寰?

        Returns:
            state: [4] numpy array
        """
        # 绌洪棿鍒╃敤鐜?
        if hasattr(self.base_env, 'space'):
            space_ratio = self.base_env.space.get_ratio()
        else:
            space_ratio = 0.0

        # 宸叉斁缃暟閲?(褰掍竴鍖?
        placed_ratio = min(self.total_placed / 50.0, 1.0)

        # 鍊欓€夌獥鍙ｅ～鍏呮瘮渚?(鍙兘涓嶆弧)
        remaining_ratio = len(self.candidate_queue) / self.window_size

        # 瀹瑰櫒鍓╀綑楂樺害
        if hasattr(self.base_env, 'space') and len(self.base_env.space.boxes) > 0:
            max_height = max([b.lz + b.z for b in self.base_env.space.boxes])
            height_ratio = 1.0 - max_height / self.container_size[2]
        else:
            height_ratio = 1.0

        return np.array([space_ratio, placed_ratio, remaining_ratio, height_ratio], dtype=np.float32)

    def step(self, selected_idx, pct_action):
        """
        Execute one step.
        """
        if self.done:
            return None, 0, True, {
                'ratio': self.base_env.space.get_ratio(),
                'total_placed': self.total_placed
            }

        placement_succeeded = False
        try:
            obs, reward, base_done, info = self.base_env.step(pct_action)
            placement_succeeded = not base_done
        except IndexError:
            obs = None
            reward = 0
            base_done = True
            info = {'error': 'EMS overflow'}

        if placement_succeeded:
            self.total_placed += 1

            base_reward = float(reward)
            flatness_bonus, flatness_score = self._compute_long_term_flatness_reward()
            reward = base_reward + flatness_bonus
            info['base_reward'] = base_reward
            info['flatness_reward'] = flatness_bonus
            info['flatness_score'] = flatness_score
            info['total_reward'] = float(reward)

            queue_list = list(self.candidate_queue)
            if len(queue_list) == 0:
                self.done = True
                ratio = self.base_env.space.get_ratio()
                info = {
                    'ratio': ratio,
                    'total_placed': self.total_placed,
                    'flatness_reward': info.get('flatness_reward', 0.0),
                    'flatness_score': info.get('flatness_score', 0.0),
                    'total_reward': info.get('total_reward', float(reward))
                }
                return obs, reward, True, info

            effective_idx = max(0, min(int(self.last_selected_idx), len(queue_list) - 1))
            queue_list.pop(effective_idx)
            self.candidate_queue = deque(queue_list, maxlen=self.window_size)

            self._fill_window()

            if len(self.candidate_queue) == 0:
                self.done = True
                ratio = self.base_env.space.get_ratio()
                info = {
                    'ratio': ratio,
                    'total_placed': self.total_placed,
                    'flatness_reward': info.get('flatness_reward', 0.0),
                    'flatness_score': info.get('flatness_score', 0.0),
                    'total_reward': info.get('total_reward', float(reward))
                }
                return obs, reward, True, info

            self._set_next_box_in_env(0)
            obs = self._generate_observation_without_gen_next_box()
            done = False
        else:
            self.done = True
            done = True
            ratio = self.base_env.space.get_ratio()
            info['ratio'] = ratio
            info['total_placed'] = self.total_placed
            info['flatness_reward'] = 0.0
            info['flatness_score'] = 0.0
            info['total_reward'] = 0.0
            reward = 0

        return obs, reward, done, info

    def get_pct_observation(self):
        return self.base_env.cur_observation()

    @property
    def observation_space(self):
        return self.base_env.observation_space

    @property
    def action_space(self):
        return self.base_env.action_space


# 娴嬭瘯浠ｇ爜
if __name__ == '__main__':
    print("SlidingWindowEnvWrapper module loaded successfully.")
    print("Trajectory-based loading with fail-to-stop semantics.")
    print("Compatible with PCT's LoadBoxCreator data format.")

