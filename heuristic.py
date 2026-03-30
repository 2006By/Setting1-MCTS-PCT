import givenData
import numpy as np
import time
from pct_envs.PctDiscrete0 import PackingDiscrete
from pct_envs.PctContinuous0 import PackingContinuous
from tools import get_args_heuristic


def _compute_height_map(space, container_size, grid_size=16):
    g = max(4, int(grid_size))
    heights = np.zeros((g, g), dtype=np.float32)
    boxes = getattr(space, 'boxes', [])
    if len(boxes) == 0:
        return heights

    width, length, _ = container_size
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


def _compute_flatness_score(space, container_size, grid_size=16):
    height_map = _compute_height_map(space, container_size, grid_size=grid_size)
    occupied = height_map > 0
    occupied_count = int(np.sum(occupied))
    if occupied_count == 0:
        return 0.0

    occ_heights = height_map[occupied]
    height_std = float(np.std(occ_heights))
    flatness_raw = 1.0 - height_std / (float(container_size[2]) + 1e-8)
    flatness_raw = float(np.clip(flatness_raw, 0.0, 1.0))

    coverage_ratio = float(occupied_count) / float(height_map.size)
    score = flatness_raw * (0.5 + 0.5 * coverage_ratio)
    return float(np.clip(score, 0.0, 1.0))


def _record_episode_metrics(env, counter, episode_utilization, episode_length,
                            episode_flatness, episode_ms_per_box, episode_start_time,
                            flatness_grid_size=16):
    result = env.space.get_ratio()
    l = len(env.space.boxes)
    flatness = _compute_flatness_score(env.space, env.bin_size, grid_size=flatness_grid_size)
    ep_elapsed = time.perf_counter() - episode_start_time
    ep_ms_per_box = ep_elapsed * 1000.0 / l if l > 0 else 0.0

    print(
        'Result of episode {}, utilization: {}, length: {}, flatness: {:.4f}, e2e(ms/box): {:.4f}'
        .format(counter, result, l, flatness, ep_ms_per_box)
    )
    episode_utilization.append(result)
    episode_length.append(l)
    episode_flatness.append(flatness)
    episode_ms_per_box.append(ep_ms_per_box)

'''
Tap-net: transportand-pack using reinforcement learning.
https://dl.acm.org/doi/abs/10.1145/3414685.3417796
'''
def MACS(env, times = 2000):
    def calc_maximal_usable_spaces(ctn, H):
        '''
        Score the given placement.
        This score function comes from https://github.com/Juzhan/TAP-Net/blob/master/tools.py
        '''
        score = 0
        for h in range(H):
            level_max_empty = 0
            # build the histogram map
            hotmap = (ctn[:, :, h] == 0).astype(int)
            histmap = np.zeros_like(hotmap).astype(int)
            for i in reversed(range(container_size[0])):
                for j in range(container_size[1]):
                    if i==container_size[0]-1: histmap[i, j] = hotmap[i, j]
                    elif hotmap[i, j] == 0: histmap[i, j] = 0
                    else: histmap[i, j] = histmap[i+1, j] + hotmap[i, j]

            # scan the histogram map
            for i in range(container_size[0]):
                for j in range(container_size[1]):
                    if histmap[i, j] == 0: continue
                    if j>0 and histmap[i, j] == histmap[i, j-1]: continue
                    # look right
                    for j2 in range(j, container_size[1]):
                        if j2 == container_size[1] - 1: break
                        if histmap[i, j2+1] < histmap[i, j]: break
                    # look left
                    for j1 in reversed(range(0, j+1)):
                        if j1 == 0: break
                        if histmap[i, j1-1] < histmap[i, j]: break
                    area = histmap[i, j] * (j2 - j1 + 1)
                    if area > level_max_empty: level_max_empty = area
            score += level_max_empty
        return score

    def update_container(ctn, pos, boxSize):
        _x, _y, _z = pos
        block_x, block_y, block_z = boxSize
        ctn[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
        under_space = ctn[_x:_x+block_x, _y:_y+block_y, 0:_z]
        ctn[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space==0 ] = -1

    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()

    container_size = env.bin_size
    container = np.zeros(env.bin_size)
    block_index = 0
    for counter in range(times):
        while True:
            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                container[:] = 0
                block_index = 0
                done = False
                break

            bestScore = -1e10
            EMS = env.space.EMS

            bestAction = None
            next_box = env.next_box
            next_den = env.next_den

            for ems in EMS:
                # Find the most suitable placement within the allowed orientation.
                for rot in range(env.orientation):
                    if rot == 0:
                        x, y, z = next_box
                    elif rot == 1:
                        y, x, z = next_box
                    elif rot == 2:
                        z, x, y = next_box
                    elif rot == 3:
                        z, y, x = next_box
                    elif rot == 4:
                        x, z, y = next_box
                    elif rot == 5:
                        y, z, x = next_box

                    if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                        for corner in range(4):
                            if corner == 0:
                                lx, ly = ems[0], ems[1]
                            elif corner == 1:
                                lx, ly = ems[3] - x, ems[1]
                            elif corner == 2:
                                lx, ly = ems[0], ems[4] - y
                            elif corner == 3:
                                lx, ly = ems[3] - x, ems[4] - y

                            # Check the feasibility of this placement
                            feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                              next_den, env.setting, returnH=True)
                            if feasible:
                                updated_containers = container.copy()
                                update_container(updated_containers, np.array([lx, ly, height]), np.array([x, y, z]))
                                score = calc_maximal_usable_spaces(updated_containers, height)

                                if score > bestScore:
                                    bestScore = score
                                    env.next_box = [x, y, z]
                                    bestAction = [0, lx, ly, height]

            if bestAction is not None:
                # Place this item in the environment with the best action.
                update_container(container, bestAction[1:4], env.next_box)
                block_index += 1
                _, _, done, _ = env.step(bestAction[0:3])
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

'''
Solving a new 3D bin packing problem with deep reinforcement learning method.
https://arxiv.org/abs/1708.05930 
'''
def LASH(env, times = 2000):

    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()
    bin_size = env.bin_size

    maxXY = [0,0]
    minXY = [bin_size[0], bin_size[1]]

    for counter in range(times):
        while True:
            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                done = False
                maxXY = [0, 0]
                minXY = [bin_size[0], bin_size[1]]
                break

            bestScore = bin_size[0] * bin_size[1] + bin_size[1] * bin_size[2] + bin_size[2] * bin_size[0]
            EMS = env.space.EMS

            bestAction = None
            next_box = env.next_box
            next_den = env.next_den

            for ems in EMS:
                # Find the most suitable placement within the allowed orientation.
                if np.sum(np.abs(ems)) == 0:
                    continue
                for rot in range(env.orientation):
                    if rot == 0:
                        x, y, z = next_box
                    elif rot == 1:
                        y, x, z = next_box
                    elif rot == 2:
                        z, x, y = next_box
                    elif rot == 3:
                        z, y, x = next_box
                    elif rot == 4:
                        x, z, y = next_box
                    elif rot == 5:
                        y, z, x = next_box

                    if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                        lx, ly = ems[0], ems[1]
                        # Check the feasibility of this placement
                        feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, returnH=True)

                        if feasible:
                            score = (max(lx + x, maxXY[0]) - min(lx, minXY[0])) * (
                                        max(ly + y, maxXY[1]) - min(ly, minXY[1])) \
                                    + (height + z) * (max(ly + y, maxXY[1]) - min(ly, minXY[1])) \
                                    + (height + z) * (max(lx + x, maxXY[0]) - min(lx, minXY[0]))

                            # The placement which keeps pack items with less surface area is better.
                            if score < bestScore:
                                bestScore = score
                                env.next_box = [x, y, z]
                                bestAction = [0, lx, ly, height, ems[3] - ems[0], ems[4] - ems[1], ems[5] - ems[2]]

                            elif score == bestScore and bestAction is not None:
                                if min(ems[3] - ems[0] - x, ems[4] - ems[1] - y, ems[5] - ems[2] - z) < \
                                        min(bestAction[4] - x, bestAction[5] - y, bestAction[6] - z):
                                    env.next_box = [x, y, z]
                                    bestAction = [0, lx, ly, height, ems[3] - ems[0], ems[4] - ems[1], ems[5] - ems[2]]
            if bestAction is not None:
                x, y, _ = env.next_box
                _, lx, ly, _, _, _, _ = bestAction
                print('bestScore: {}, bestAction:{}'.format(bestScore, bestAction))
                print('lx: {}, ly: {}'.format(lx, ly))
                if lx + x > maxXY[0]: maxXY[0] = lx + x
                if ly + y > maxXY[1]: maxXY[1] = ly + y
                if lx < minXY[0]: minXY[0] = lx
                if ly < minXY[1]: minXY[1] = ly
                # Place this item in the environment with the best action.
                _, _, done, _ = env.step(bestAction[0:3])
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

'''
Stable bin packing of non-convex 3D objects with a robot manipulator.
https://doi.org/10.1109/ICRA.2019.8794049
'''
def heightmap_min(env, times = 2000):
    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()
    bin_size = env.bin_size

    for counter in range(times):
        while True:
            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                done = False
                break

            bestScore = 1e10
            bestAction = []

            next_box = env.next_box
            next_den = env.next_den

            for lx in range(bin_size[0] - next_box[0] + 1):
                for ly in range(bin_size[1] - next_box[1] + 1):
                    # Find the most suitable placement within the allowed orientation.
                    for rot in range(env.orientation):
                        if rot == 0:
                            x, y, z = next_box
                        elif rot == 1:
                            y, x, z = next_box
                        elif rot == 2:
                            z, x, y = next_box
                        elif rot == 3:
                            z, y, x = next_box
                        elif rot == 4:
                            x, z, y = next_box
                        elif rot == 5:
                            y, z, x = next_box

                        # Check the feasibility of this placement
                        feasible, heightMap = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                             next_den, env.setting, False, True)
                        if not feasible:
                            continue

                        # Score the given placement.
                        score = lx + ly + 100 * np.sum(heightMap)
                        if score < bestScore:
                            bestScore = score
                            env.next_box = [x, y, z]
                            bestAction = [0, lx, ly]

            if len(bestAction) != 0:
                # Place this item in the environment with the best action.
                env.step(bestAction)
                done = False
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

'''
Randomly pick placements from full coordinates.
'''
def random(env, times = 2000):
    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()
    bin_size = env.bin_size

    for counter in range(times):
        while True:
            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                done = False
                break

            next_box = env.next_box
            next_den = env.next_den

            # Check the feasibility of all placements.
            candidates = []
            for lx in range(bin_size[0] - next_box[0] + 1):
                for ly in range(bin_size[1] - next_box[1] + 1):
                    for rot in range(env.orientation):
                        if rot == 0:
                            x, y, z = next_box
                        elif rot == 1:
                            y, x, z = next_box
                        elif rot == 2:
                            z, x, y = next_box
                        elif rot == 3:
                            z, y, x = next_box
                        elif rot == 4:
                            x, z, y = next_box
                        elif rot == 5:
                            y, z, x = next_box

                        feasible, heightMap = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                         next_den, env.setting, False, True)
                        if not feasible:
                            continue

                        candidates.append([[x, y, z], [0, lx, ly]])

            if len(candidates) != 0:
                # Pick one placement randomly from all possible placements
                idx = np.random.randint(0, len(candidates))
                env.next_box = candidates[idx][0]
                env.step(candidates[idx][1])
                done = False
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

'''
An Online Packing Heuristic for the Three-Dimensional Container Loading
Problem in Dynamic Environments and the Physical Internet
https://doi.org/10.1007/978-3-319-55792-2\_10
'''
def OnlineBPH(env, times = 2000):
    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()

    for counter in range(times):
        while True:
            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                done = False
                break

            # Sort the ems placement with deep-bottom-left order.
            EMS = env.space.EMS
            EMS = sorted(EMS, key=lambda ems: (ems[2], ems[1], ems[0]), reverse=False)

            bestAction = None
            next_box = env.next_box
            next_den = env.next_den
            stop = False


            for ems in EMS:
                # Find the first suitable placement within the allowed orientation.
                if np.sum(np.abs(ems)) == 0:
                    continue
                for rot in range(env.orientation):
                    if rot == 0:
                        x, y, z = next_box
                    elif rot == 1:
                        y, x, z = next_box
                    elif rot == 2:
                        z, x, y = next_box
                    elif rot == 3:
                        z, y, x = next_box
                    elif rot == 4:
                        x, z, y = next_box
                    elif rot == 5:
                        y, z, x = next_box

                    # Check the feasibility of this placement
                    if env.space.drop_box_virtual([x, y, z], (ems[0], ems[1]), False, next_den, env.setting):
                        env.next_box = [x, y, z]
                        bestAction = [0, ems[0], ems[1]]
                        stop = True
                        break
                if stop: break

            if bestAction is not None:
                # Place this item in the environment with the best action.
                _, _, done, _ = env.step(bestAction)
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

'''
A Hybrid Genetic Algorithm for Packing in 3D with Deepest Bottom Left with Fill Method
https://doi.org/10.1007/978-3-540-30198-1\_45
'''
def DBL(env, times = 2000):
    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()
    bin_size = env.bin_size

    for counter in range(times):
        while True:
            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                done = False
                break

            bestScore = 1e10
            bestAction = []

            next_box = env.next_box
            next_den = env.next_den

            for lx in range(bin_size[0] - next_box[0] + 1):
                for ly in range(bin_size[1] - next_box[1] + 1):
                    # Find the most suitable placement within the allowed orientation.
                    for rot in range(env.orientation):
                        if rot == 0:
                            x, y, z = next_box
                        elif rot == 1:
                            y, x, z = next_box
                        elif rot == 2:
                            z, x, y = next_box
                        elif rot == 3:
                            z, y, x = next_box
                        elif rot == 4:
                            x, z, y = next_box
                        elif rot == 5:
                            y, z, x = next_box

                        # Check the feasibility of this placement
                        feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                         next_den, env.setting, True, False)
                        if not feasible:
                            continue

                        # Score the given placement.
                        score = lx + ly + 100 * height
                        if score < bestScore:
                            bestScore = score
                            env.next_box = [x, y, z]
                            bestAction = [0, lx, ly]

            if len(bestAction) != 0:
                # Place this item in the environment with the best action.
                env.step(bestAction)
                done = False
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

'''
Online 3D Bin Packing with Constrained Deep Reinforcement Learning
https://ojs.aaai.org/index.php/AAAI/article/view/16155
'''
def BR(env, times = 2000):
    def eval_ems(ems):
        # Score the given placement.
        s = 0
        valid = []
        for bs in env.item_set:
            bx, by, bz = bs
            if ems[3] - ems[0] >= bx and ems[4] - ems[1] >= by and ems[5] - ems[2] >= bz:
                valid.append(1)
        s += (ems[3] - ems[0]) * (ems[4] - ems[1]) * (ems[5] - ems[2])
        s += len(valid)
        if len(valid) == len(env.item_set):
            s += 10
        return s

    done = False
    episode_utilization = []
    episode_length = []
    episode_flatness = []
    episode_ms_per_box = []
    env.reset()
    episode_start_time = time.perf_counter()

    for counter in range(times):
        while True:

            if done:
                # Reset the enviroment when the episode is done
                _record_episode_metrics(
                    env, counter, episode_utilization, episode_length,
                    episode_flatness, episode_ms_per_box, episode_start_time
                )
                env.reset()
                episode_start_time = time.perf_counter()
                done = False
                break
            

            bestScore = -1e10
            EMS = env.space.EMS

            bestAction = None
            next_box = env.next_box
            next_den = env.next_den

            for ems in EMS:
                # Find the most suitable placement within the allowed orientation.
                for rot in range(env.orientation):
                    if rot == 0:
                        x, y, z = next_box
                    elif rot == 1:
                        y, x, z = next_box
                    elif rot == 2:
                        z, x, y = next_box
                    elif rot == 3:
                        z, y, x = next_box
                    elif rot == 4:
                        x, z, y = next_box
                    elif rot == 5:
                        y, z, x = next_box

                    if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                        lx, ly = ems[0], ems[1]
                        # Check the feasibility of this placement
                        feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                          next_den, env.setting, returnH=True)
                        if feasible:
                            score = eval_ems(ems)
                            if score > bestScore:
                                bestScore = score
                                env.next_box = [x, y, z]
                                bestAction = [0, lx, ly, height]


            if bestAction is not None:
                # Place this item in the environment with the best action.
                _, _, done, _ = env.step(bestAction[0:3])
            else:
                # No feasible placement, this episode is done.
                done = True

    return {
        'ratios': episode_utilization,
        'placed': episode_length,
        'flatness_scores': episode_flatness,
        'e2e_ms_per_box_episode': episode_ms_per_box
    }

if __name__ == '__main__':
    args = get_args_heuristic()

    if args.continuous == True: PackingEnv = PackingContinuous
    else: PackingEnv = PackingDiscrete

    env = PackingEnv(setting = args.setting,
                     container_size = args.container_size,
                     item_set = args.item_size_set,
                     data_name = args.dataset_path,
                     load_test_data = args.load_dataset,
                     internal_node_holder = 80,
                     leaf_node_holder = 1000)

    eval_start = time.perf_counter()

    if args.heuristic == 'LSAH':
        results = LASH(env, args.evaluation_episodes)
    elif args.heuristic == 'MACS':
        results = MACS(env, args.evaluation_episodes)
    elif args.heuristic == 'HM':
        results = heightmap_min(env, args.evaluation_episodes)
    elif args.heuristic == 'RANDOM':
        results = random(env, args.evaluation_episodes)
    elif args.heuristic == 'OnlineBPH':
        results = OnlineBPH(env, args.evaluation_episodes)
    elif args.heuristic == 'DBL':
        results = DBL(env, args.evaluation_episodes)
    elif args.heuristic == 'BR':
        results = BR(env, args.evaluation_episodes)
    else:
        raise ValueError(f"Unknown heuristic: {args.heuristic}")

    elapsed_s = time.perf_counter() - eval_start
    ratios = np.asarray(results.get('ratios', []), dtype=np.float64)
    placed = np.asarray(results.get('placed', []), dtype=np.float64)
    flatness_scores = np.asarray(results.get('flatness_scores', []), dtype=np.float64)
    e2e_ms_ep = np.asarray(results.get('e2e_ms_per_box_episode', []), dtype=np.float64)

    if ratios.size == 0:
        raise RuntimeError("No episode results were collected.")

    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    var_ratio = float(np.var(ratios))
    max_ratio = float(np.max(ratios))
    min_ratio = float(np.min(ratios))

    mean_placed = float(np.mean(placed)) if placed.size > 0 else 0.0
    std_placed = float(np.std(placed)) if placed.size > 0 else 0.0
    var_placed = float(np.var(placed)) if placed.size > 0 else 0.0

    mean_flatness = float(np.mean(flatness_scores)) if flatness_scores.size > 0 else 0.0
    std_flatness = float(np.std(flatness_scores)) if flatness_scores.size > 0 else 0.0
    var_flatness = float(np.var(flatness_scores)) if flatness_scores.size > 0 else 0.0
    max_flatness = float(np.max(flatness_scores)) if flatness_scores.size > 0 else 0.0
    min_flatness = float(np.min(flatness_scores)) if flatness_scores.size > 0 else 0.0

    total_boxes = float(np.sum(placed)) if placed.size > 0 else 0.0
    e2e_ms_global = elapsed_s * 1000.0 / total_boxes if total_boxes > 0 else 0.0
    e2e_ms_mean_episode = float(np.mean(e2e_ms_ep)) if e2e_ms_ep.size > 0 else 0.0
    e2e_ms_std_episode = float(np.std(e2e_ms_ep)) if e2e_ms_ep.size > 0 else 0.0
    e2e_ms_var_episode = float(np.var(e2e_ms_ep)) if e2e_ms_ep.size > 0 else 0.0

    print("\n" + "="*50)
    print(f"EVALUATION RESULTS (Heuristic-{args.heuristic})")
    print("="*50)
    print(f"Heuristic: {args.heuristic}")
    print(f"Test Data: {args.dataset_path}")
    print(f"Episodes: {len(ratios)}")
    print(f"Setting: {args.setting} | Continuous: {args.continuous}")
    print("Space Utilization:")
    print(f"  Mean: {mean_ratio:.4f} ({mean_ratio*100:.2f}%)")
    print(f"  Std:  {std_ratio:.4f}")
    print(f"  Var:  {var_ratio:.6f}")
    print(f"  Max:  {max_ratio:.4f} ({max_ratio*100:.2f}%)")
    print(f"  Min:  {min_ratio:.4f} ({min_ratio*100:.2f}%)")
    print("Boxes Placed:")
    print(f"  Mean: {mean_placed:.1f}")
    print(f"  Std:  {std_placed:.4f}")
    print(f"  Var:  {var_placed:.4f}")
    print("Packing Flatness:")
    print(f"  Mean: {mean_flatness:.4f}")
    print(f"  Std:  {std_flatness:.4f}")
    print(f"  Var:  {var_flatness:.6f}")
    print(f"  Max:  {max_flatness:.4f}")
    print(f"  Min:  {min_flatness:.4f}")
    print("Timing:")
    print("  Model inference total: N/A (heuristic search)")
    print("  Model inference mean:  N/A")
    print("  Model inference std:   N/A")
    print("  Model inference var:   N/A")
    print(f"  End-to-end total: {elapsed_s:.4f}s")
    print(f"  End-to-end mean(global): {e2e_ms_global:.4f} ms/box")
    print(f"  End-to-end mean(episode): {e2e_ms_mean_episode:.4f} ms/box")
    print(f"  End-to-end std(episode):  {e2e_ms_std_episode:.4f} ms/box")
    print(f"  End-to-end var(episode):  {e2e_ms_var_episode:.6f} (ms/box)^2")
    print("="*50)
