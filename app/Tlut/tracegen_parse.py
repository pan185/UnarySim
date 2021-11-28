import utils

class Prob(object):
    """Problem space with layer dimension, stride and dilation defined.
    
    Attributes: 
        prob: A layer dimemsion dictionary. 
            R, S represent the weight filter width and height.
            P, Q represent the output feature map width and height.
            C represents the input channel size. 
            K represents the output channel size.
            N represents the batch size. 
            Wstride, Hstride represent the width and height dimension stride.
            Wdilation, Hdilation represent the width and height dimension dilation.
        prob_bound:  A 1d array with layer dimension value for R,S,P,Q,C,K,N
            e.g. [1,1,1,2,3,4,5]
        prob_factors:  A 2d array with all prime factors generated from each dimension
            e.g. [[1],[1],[1],[2],[3],[2,2],[5]] 
    """

    def __init__(self, prob_path):
        """Initialize the layer dimension from an input yaml file. 

            Example input yaml file format: 
                problem:
                  C: 3
                  Hdilation: 1
                  Hstride: 2
                  K: 64
                  N: 1
                  P: 112
                  Q: 112
                  R: 7
                  S: 7
                  Wdilation: 1
                  Wstride: 2
                  shape: cnn-layer


        Args: 
            prob_path: Path to the yaml file that defines the convolution layer dimensions. 
        """
        # defines the dimension index for 7 major loop bounds 
        self.prob_idx_name_dict = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}
        self.prob_name_idx_dict = {v: k for k, v in self.prob_idx_name_dict.items()}

        self.prob_bound = [-1] * len(self.prob_name_idx_dict)
        # self.prob_factors = []
        # for i in range(len(self.prob_name_idx_dict)):
        #     self.prob_factors.append([])

        self.prob_levels = len(self.prob_idx_name_dict.items())

        self.path = prob_path.resolve()
        prob_dict = utils.parse_yaml(self.path)
        self.prob = prob_dict['problem']

        for key, value in self.prob.items():
            if ('stride' in key or 'dilation' in key):
                continue
            if (key == 'shape'):
                continue
            if (key == 'et_cycle'):
                continue
            prob_idx = self.prob_name_idx_dict[key]
            self.prob_bound[prob_idx] = value
            # self.prob_factors[prob_idx] = utils.get_prime_factors(value)

    def config_str(self):
        """Returns the key str name for representing a unique layer."""
        # val_arr = []
        # for value in self.prob_bound:
        #     val_arr.append(str(value))
        # keys = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']
        # val_arr.extend([str(self.prob[key]) for key in keys])
        # val_str = "_".join(val_arr)
        # return val_str
        return self.path.stem

    def print(self):
        print(self.__dict__)
    
class Arch(object):
    """ Hardware architecture specifyng number of hardware instances and buffer capacity.
    
    Attributes: 
        mem_instances: number of memory instances per chip.
        mem_entries: number of valid memory entries.
    """

    def __init__(self, arch_path):
        """Initialize the hardware architecture details from an input yaml file. 

        Args: 
            arch_path: Path to the yaml file that defines the hardware architecture constraints. 
        """

        self.path = arch_path.resolve()
        arch_dict = utils.parse_yaml(self.path)

        # arch config version, please add a postfix _v3 to
        # the yaml filename if a new version is used
        version = 'v3' if '_v3' in self.path.name else 'v0'

        # mem instance size for each 
        self.mem_instances = []
        self.mem_entries = []
        self.mem_bw = []

        # name to idx lookup
        self.mem_idx = {}

        # idx to name lookup
        self.mem_name = {}

        if version == 'v0':
            self.arch = arch_dict['arch']
            for key, value in self.arch.items():
                setattr(self, key, value)
            for i, mem in enumerate(self.storage):
                self.mem_idx[mem['name']] = i
                self.mem_name[i] = mem['name']
                self.mem_instances.append(mem['instances'])
                if i < len(self.storage) - 1:
                    self.mem_entries.append(mem['entries'])
                    self.mem_bw.append(mem['bw'])
        elif version == 'v3':
            assert(False)
            self.dram = arch_dict['architecture']['subtree'][0]['local'][0]
            self.global_buf = arch_dict['architecture']['subtree'][0]['subtree'][0]['local'][0]
            self.pe_buf = arch_dict['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local']
            idx = 0
            for i, mem in enumerate(self.pe_buf[::-1]):
                if mem['class'] == 'SRAM' or mem['class'] == 'regfile':
                    self.mem_idx[mem['name']] = idx
                    self.mem_name[idx] = mem['name']
                    self.mem_instances.append(mem['attributes']['instances'])
                    self.mem_entries.append(mem['attributes']['entries'])
                    idx += 1

            self.mem_idx[self.global_buf['name']] = idx
            self.mem_name[idx] = self.global_buf['name']
            self.mem_instances.append(self.global_buf['attributes']['instances'])
            self.mem_entries.append(self.global_buf['attributes']['entries'])
            idx += 1
            self.mem_idx[self.dram['name']] = idx
            self.mem_name[idx] = self.dram['name']
            self.mem_instances.append(self.dram['attributes']['instances'])
            self.arch = {"instances": self.mem_instances, "entries": self.mem_entries}
        self.mem_levels = len(self.mem_idx.items())
        self.S = self.gen_spatial_constraint()

    def gen_spatial_constraint(self):
        """Generate spatial constraints."""
        S = []
        inner_instances = self.mem_instances[0]
        for i in self.mem_instances:
            if i != 0:
                S.append(inner_instances // i)
                inner_instances = i
        return S

    def config_str(self):
        """Return the filename for the input yaml with postfix."""
        return self.path.stem

    def print(self):
        print(self.__dict__)

class Dataflow(object):
    """
    Dataflow object that describes the mapping from compute to hardware

    Attributes:
        type: OutputStationary/InputStationary/WeightStationary
        TileStationary: I/W
    """

    def __init__(self, dtf_path):
        self.path = dtf_path.resolve()
        dtf_dict = utils.parse_yaml(self.path)

        self.type = dtf_dict['Type']
        self.tileStationary = dtf_dict['TileStationary']

    def config_str(self):
        """Return the filename for the input yaml with postfix."""
        return self.path.stem

    def print(self):
        print(self.__dict__)