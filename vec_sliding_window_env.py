import multiprocessing as mp
import numpy as np
import cloudpickle
import pickle
import os
import contextlib

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    Clear MPI environment variables to prevent hangs when starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

def _subproc_worker(pipe, parent_pipe, env_fn_wrapper):
    """
    Control a single environment instance using IPC.
    """
    # 子进程需要独立注册环境，因为 spawn 模式下不继承 main process 的 gym registry
    from tools import registration_envs
    registration_envs()
    
    env = env_fn_wrapper.x()
    parent_pipe.close()
    
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                obs = env.reset()
                pipe.send((obs, env.get_candidates(), env.get_container_state()))
            elif cmd == 'set_selected_and_get_obs':
                obs = env.set_selected_and_get_obs(data)
                pipe.send(obs)
            elif cmd == 'step':
                selected_idx, pct_action = data
                obs, reward, done, info = env.step(selected_idx, pct_action)
                if done:
                    obs = env.reset()
                pipe.send((obs, reward, done, info, env.get_candidates(), env.get_container_state()))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError(f'Got unrecognized cmd {cmd}')
    except KeyboardInterrupt:
        print('VecSlidingWindowEnv worker: got KeyboardInterrupt')
    finally:
        pass


class VecSlidingWindowEnv:
    """
    A vectorized wrapper for SlidingWindowEnvWrapper using multiprocessing.
    """
    def __init__(self, env_fns, context='spawn'):
        ctx = mp.get_context(context)
        self.num_envs = len(env_fns)
        self.parent_pipes, self.child_pipes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, child_pipe, parent_pipe in zip(env_fns, self.child_pipes, self.parent_pipes):
                wrapped_fn = CloudpickleWrapper(env_fn)
                proc = ctx.Process(target=_subproc_worker, args=(child_pipe, parent_pipe, wrapped_fn))
                proc.daemon = True
                proc.start()
                self.procs.append(proc)
                child_pipe.close()

    def reset(self):
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        results = [pipe.recv() for pipe in self.parent_pipes]
        obs, candidates, container_states = zip(*results)
        return np.array(obs), np.array(candidates), np.array(container_states)

    def set_selected_and_get_obs(self, selected_indices):
        assert len(selected_indices) == self.num_envs
        for pipe, idx in zip(self.parent_pipes, selected_indices):
            # item() converts numpy int/float to python native scalar for safe IPC
            pipe.send(('set_selected_and_get_obs', idx.item()))
        return np.array([pipe.recv() for pipe in self.parent_pipes])

    def step(self, selected_indices, pct_actions):
        assert len(selected_indices) == self.num_envs
        assert len(pct_actions) == self.num_envs
        for pipe, idx, act in zip(self.parent_pipes, selected_indices, pct_actions):
            pipe.send(('step', (idx.item(), act)))
        results = [pipe.recv() for pipe in self.parent_pipes]
        obs, rewards, dones, infos, candidates, container_states = zip(*results)
        return (np.array(obs), np.array(rewards), np.array(dones), 
                infos, np.array(candidates), np.array(container_states))

    def close(self):
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()
