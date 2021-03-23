import numpy as np
from pathlib import Path
import itertools
import torch
import random
import core as core
from typing import Optional, Any, Tuple, Union, Dict,List
import sys


class Logger:
    """
    Logger class to log important information
    :param logdir: Directory to save log at
    :param formats: Formatting of each log ['csv', 'stdout', 'tensorboard']
    :type logdir: string
    :type formats: list
    """

    def __init__(self, logdir: str = None, formats: List[str] = ["csv"]):
        if logdir is None:
            self._logdir = os.getcwd()
        else:
            self._logdir = logdir
            if not os.path.isdir(self._logdir):
                os.makedirs(self._logdir)
        self._formats = formats
        self.writers = []
        for ft in self.formats:
            self.writers.append(get_logger_by_name(ft)(self.logdir))

    def write(self, kvs: Dict[str, Any], log_key: str = "timestep") -> None:
        """
        Add entry to logger
        :param kvs: Entry to be logged
        :param log_key: Key plotted on log_key
        :type kvs: dict
        :type log_key: str
        """
        for writer in self.writers:
            writer.write(kvs, log_key)

    def close(self) -> None:
        """
        Close the logger
        """
        for writer in self.writers:
            writer.close()

    @property
    def logdir(self) -> str:
        """
        Return log directory
        """
        return self._logdir

    @property
    def formats(self) -> List[str]:
        """
        Return save format(s)
        """
        return self._formats
    
    
class logger:
    """
    A general-purpose logger.
    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.
        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 
            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if proc_id()==0:
            self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id()==0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.
        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id()==0:
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.
        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 
        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.
        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.
            itr: An int, or None. Current iteration of training.
        """
        if proc_id()==0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle state_dict.', color='red')
            if hasattr(self, 'tf_saver_elements'):
                self._tf_simple_save(itr)
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)
                
    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.
        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.
        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id()==0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = osp.join(self.output_dir, fpath)
            fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do 
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.
                torch.save(self.pytorch_saver_elements, fname)
                
def set_seeds(seed: int, env = None) -> None:
    """
    Sets seeds for reproducibility
    :param seed: Seed Value
    :param env: Optionally pass gym environment to set its seed
    :type seed: int
    :type env: Gym Environment
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        
        
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


                
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        
        #adding extra initializations for normalization
        self.obs_buf_max = np.zeros((self.obs_buf.shape[1],1),dtype=np.float32)
        self.obs2_buf_max = np.zeros((self.obs2_buf.shape[1],1),dtype=np.float32)
        self.rew_buf_max = np.zeros((1,1),dtype=np.float32)
        
        self.obs_buf_min = np.zeros((self.obs_buf.shape[1],1),dtype=np.float32)
        self.obs2_buf_min = np.zeros((self.obs2_buf.shape[1],1),dtype=np.float32)
        self.rew_buf_min = np.zeros((1,1),dtype=np.float32)
        
        #adding extra initialization for adding rewards
        #self.past_rew = np.zeros((4,1),dtype=np.float32)
        

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
        
        
        '''
        Ideas for storing the reward of the last 4 time steps:
        and adding it to state space.
        
        Either take the current time step and directly slice t-4 from it.Normalize them using the current min max.
        Use these to calculate penalties that are used to calculate objective function using the same formula and normalize them.
        
        For reward function store the previous time step reward and calculate how much it is contributing to ramping.
        Take care of the difference reward function.We cannot store more than one of the previous rewrd function.
        
        Add actions and corresponding SOC's to tensorboard to monitor if it is chooses actions even when the SOC is 100%.
        Add weights and gradients to tensorboard.
        
        Once you know that you are normalizing rewards and it helps for learning across environments in which reward vary in the large ranges
        train parallely on the 4 environments, although it is not much realizable in the practical scenario, one can try it.
        
        '''
        
    def collect_minmax(self):
        # if self.obs_buf.shape
        
        #print('Collecting min max values for buffer')
        #print(self.obs_buf[8459,:])
        self.obs_buf_max = np.max(self.obs_buf,axis=0)
        self.obs2_buf_max = np.max(self.obs2_buf,axis=0)
        self.rew_buf_max = np.max(self.rew_buf)
        
        self.obs_buf_min = np.min(self.obs_buf,axis=0)
        self.obs2_buf_min = np.min(self.obs2_buf,axis=0)
        self.rew_buf_min = np.min(self.rew_buf)
        
    def sample_batch(self, batch_size=32):
        
        #add normalization routine here after sampling and before converting them to torch tensors.
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        raw_obs=self.obs_buf[idxs]
        raw_obs2=self.obs2_buf[idxs]
        #act=self.act_buf[idxs]
        raw_rew=self.rew_buf[idxs]
        #done=self.done_buf[idxs]
        
        batch_obs_max = np.max(raw_obs,axis=0)
        batch_obs2_max = np.max(raw_obs2,axis=0)
        batch_rew_max = np.max(raw_rew,axis=0)
        
        batch_obs_min = np.min(raw_obs,axis=0)
        batch_obs2_min = np.min(raw_obs2,axis=0)
        batch_rew_min = np.min(raw_rew,axis=0)
        
        #updating buffer minmax
        #print(batch_rew_max)
        #print(batch_rew_min)
        
        #print('main rewards')
        #print(self.rew_buf_max)
        #print(self.rew_buf_min)
        if batch_rew_max > self.rew_buf_max :
            self.rew_buf_max = batch_rew_max
            
        if batch_rew_min < self.rew_buf_min :
            self.rew_buf_min = batch_rew_min
        
        
        #print('printing buffer observations')
        #print(self.obs_buf_max)
        #print(self.obs2_buf_max)
        
        for i in range(batch_obs_max.size):
            if batch_obs_max[i] > self.obs_buf_max[i] :
                self.obs_buf_max[i] = batch_obs_max[i]
                
            if batch_obs2_max[i] > self.obs2_buf_max[i] :
                self.obs2_buf_max[i] = batch_obs2_max[i]
                                
            if batch_obs_min[i] < self.obs_buf_min[i] :
                self.obs_buf_min[i] = batch_obs_min[i]
                
            if batch_obs2_min[i] < self.obs2_buf_min[i] :
                self.obs2_buf_min[i] = batch_obs2_min[i]
            
        #normalizing observations and rewards
        
        #norm_obs = (raw_obs - self.obs_buf_min) / (self.obs_buf_max - self.obs_buf_min + 1e-6 )
        #norm_obs2 = (raw_obs2 - self.obs2_buf_min) / (self.obs2_buf_max - self.obs2_buf_min + 1e-6 )
        #norm_rew = (raw_rew - self.rew_buf_min) /(self.rew_buf_max - self.rew_buf_min + 1e-6 )
        
        # for reward to be within the range -1 to 1
        # https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy  
        def normalize_samples(self,x_min,x_max,to_normalize):
            if to_normalize== 'raw_obs' :
                nom =(raw_obs - self.obs_buf_min)*(x_max-x_min)
                denom = (self.obs_buf_max - self.obs_buf_min)
                #denom = denom + (denom is 0)
                denom[denom==0] = 1
                return  x_min + nom/denom 
            if to_normalize== 'raw_obs2' :
                nom =(raw_obs2 - self.obs2_buf_min)*(x_max-x_min)
                denom = (self.obs2_buf_max - self.obs2_buf_min)
                #denom = denom + (denom is 0)
                denom[denom==0] = 1
                return  x_min + nom/denom 
            if to_normalize== 'raw_rew' :
                nom =(raw_rew - self.rew_buf_min)*(x_max-x_min)
                denom = self.rew_buf_max - self.rew_buf_min
                denom = denom + (denom is 0)
                #denom[denom==0] = 1
                return  x_min + nom/denom 
                
        norm_obs = normalize_samples(self,0,1,to_normalize='raw_obs')
        norm_obs2 = normalize_samples(self,0,1,to_normalize='raw_obs2')
        norm_rew = normalize_samples(self,-1,1,to_normalize='raw_rew')
        
        batch = dict(obs=norm_obs,
                     obs2=norm_obs2,
                     act=self.act_buf[idxs],
                     rew=norm_rew,
                     done=self.done_buf[idxs])
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
