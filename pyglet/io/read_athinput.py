import re
import collections

def read_athinput(filename, as_namedtuple=False, verbose=False):
    """
    Function to read athinput from simulation log
    
    Parameters
    ----------
    filename : str
        Name of the file to open, including extension
    verbose : bool
        Print verbose message
    
    Returns
    -------
    par : dict or namedtuple
        Each item is a dictionary or namedtuple containing individual input 
        block.
    """

    if verbose:
        print('[read_par]: Reading params from {0}'.format(filename))

    istart = 0
    iend = 0
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    PAR_DUMP = False
    for i, line in enumerate(lines):
        if 'PAR_DUMP' in line:
            PAR_DUMP = True
            break

    if PAR_DUMP:
        flag = True
        for i, line in enumerate(lines):
            if 'PAR_DUMP' in line:
                if flag:
                    istart = i
                    flag = False
                else:
                    iend = i
    else:
        for i, line in enumerate(lines):
            if '<comment>' in line:
                istart = i
            if '<par_end>' in line:
                iend = i

    if iend == 0:
        lines = lines[istart:]
    else:
        lines = lines[istart:iend]
    
    # Parse lines
    reblock = re.compile(r"<\w+>\s*")
    ##  reparam=re.compile(r"[-\[\]\w]+\s*=")
    # To deal with space (such as star particles in <configure>)
    reparam = re.compile(r"[-\[\]\w]+[\s*[-\[\]\w]*]*\s*=")

    # Find blocks first
    block = []
    for l in lines:
        b = reblock.match(l)
        if b is not None:
            block.append(b.group().strip()[1:-1])

    # remove comment block
    block.remove('comment')

    o = {}
    for b in block:
        o.setdefault(b, {})

    # Add keys and values to each block
    for l in lines:
        b = reblock.match(l)
        p = reparam.match(l)
        if b is not None:
            bstr = b.group().strip()[1:-1]
            if bstr in o:
                bname = bstr # bname is valid block
            else:
                bname = None
        elif p is not None and bname is not None:
            lsplit = l.split()
            i1_found=False
            for i, lsplit_ in enumerate(lsplit):
                if lsplit_ == '=':
                    i0 = i
                    break

            pname = '_'.join(lsplit[:i0])
            value = lsplit[i0+1]

            # Evaluate if value is floating point number (or real number) or
            # integer or string
            if re.match(r'^[+-]?\d*\.\d*[eE][+-]?\d+$',value) or \
               re.match(r'^[+-]?\d+[eE][+-]?\d+$',value) or \
               re.match(r'^[+-]?\d+\.\d*$',value):
                o[bname][pname] = float(value)
            elif re.match(r'^-?[0-9]+$',value):
                o[bname][pname] = int(value)
                #o[bname][l.split()[0]]=float(value)
            elif value.lower() == 'true' or  value.lower() == 'false' or \
                 value.lower() == 'none':
                o[bname][pname] = eval(value.capitalize())
            else:
                o[bname][pname] = value

    if as_namedtuple:
        # Convert to namedtuple
        par = collections.namedtuple('par', o.keys())(**o)
        return par
    else:
        return o

