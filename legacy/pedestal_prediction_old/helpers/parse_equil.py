def mhd_load(self,mhd_loc,fp):
    """Load and calculate various MHD equilibrium parameters using method specified by mhd_loc flag. 
    
    Parameters
    ----------
    mhd_loc : string
        which method to use to load MHD parameters.
    fp : string
        filepath to file with MHD parameters.
            
    """

    if mhd_loc == 'eqdsk':
        from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
        return read_eqdsk(fp)