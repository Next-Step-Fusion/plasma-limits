import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import simpson
from scipy import constants
import os


class epednn_class:
    """Mixin adding the EPEDNN methods
    """
    def calc_volavgP(self,pres_gfile=False):
        """Calculate the volume-averaged pressure

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        pres_gfile : boolean
            if True, use the pressure from the g-file

        Sets
        ----
        self.volavgP : float
            Volume-averaged pressure (same units as self.pres).
        """

        psi_N_plasma = self.psi_N_pres
        if pres_gfile: 
            pressure = self.eq['pres'][1:]
        else:
            n_i_pres = interp1d(self.psi_ni_eval,self.n_i, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)
            n_e_pres = interp1d(self.psi_ne_eval,self.n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)
            T_i_pres = interp1d(self.psi_Ti_eval,self.T_i, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma) * 1e3 # keV -> eV
            T_e_pres = interp1d(self.psi_Te_eval,self.T_e, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma) * 1e3 # keV -> eV

            pressure = (n_i_pres * T_i_pres + n_e_pres * T_e_pres) * constants.e # Pa

        # Volume average the pressure
        V_full_plasma = interp1d(self.psi_N_pres, self.V_plasma, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_N_plasma)
        dV_dpsi = np.gradient(V_full_plasma, psi_N_plasma)
        self.volavgP = (simpson(pressure * dV_dpsi, psi_N_plasma)
                        / simpson(dV_dpsi, psi_N_plasma))
                
    def calc_betan(self,pres_gfile=False):
        """Calculate the normalized beta

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        -------
        self.Betan : float
            Normalized beta, dimensionless
        """

        self.calc_volavgP(pres_gfile)

        # _, [B_R, B_Z, _] = xself.calc_B(self.eq['rzout'][:, 0], self.eq['rzout'][:, 1])
        # bp_lcfs = np.sqrt(B_R**2 + B_Z**2)
        # bp_avg = np.mean(bp_lcfs)

        betat = self.volavgP / (self.bt**2 / (2 * self.mu0))
        self.betat = betat
        # betap = self.volavgP / (bp_avg**2 / (2 * self.mu0))
        # beta = ((1/betat) + (1/betap))**(-1)

        # EPED / Troyon: β_N = β_t[%] * a * abs(B_t) [T] / I_p[MA] -> this is what OpenFUSIONToolkit uses for β_N
        if self.verbose:
            print(f'betat: {betat}, a: {self.a}, bt: {self.bt}, Ip: {self.Ip}')
        self.betan = 100 * betat * (self.a * abs(self.bt) / abs(self.Ip))

    def setup_epednn(self, model='EPED1'):
        """Setup the EPEDNN model with quantities from the Saarelma-Connor setup

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Returns
        -------
        pedestal_pressure : float
            Pedestal pressure (MPa)
        pedestal_width : float
            Pedestal width (normalized poloidal flux)
        """

        print("Setting up EPEDNN...")

        if model == 'EPED1':
            # Requires dependency "juliacall" to translate Python inputs to FUSE EPED.jl
            # Requires dependency EPEDNN

            import juliapkg

            # 1. Tell juliapkg to add your local EPEDNN package in development mode.
            # This registers it with the isolated Julia environment PythonCall uses.
            epednn_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), # ESCAPE root (src/epednn/..)
                "dependencies",
                "EPEDNN.jl",
            )
            if not os.path.isdir(epednn_path):
                raise FileNotFoundError(
                    f"EPEDNN.jl not found at {epednn_path}. "
                    "Initialize it with: git submodule update --init dependencies/EPEDNN.jl"
                )
            juliapkg.add(
                "EPEDNN",
                uuid="e64856f0-3bb8-4376-b4b7-c03396503991",
                path=epednn_path,
                dev=True,
            )

            # 2. Resolve and instantiate. THIS is what fixes your missing dependency error!
            juliapkg.resolve()

            # 3. Now that the environment is set up, load juliacall and your package
            from juliacall import Main as jl
            jl.seval('using EPEDNN')

            # from juliacall import Main as jl

            # 1. Load the Julia EPEDNN module (Assuming EPEDNN is already installed in your Julia environment)
            '''
                # To install EPEDNN, run the following command in your terminal:
                conda activate sc_ped
                cd /Users/nelsonlab/codes/saarelma-conner-ped
                julia

                # if you need to install julia, run the following command in your terminal:
                curl -fsSL https://install.julialang.org | sh
                # then restart terminal
                julia --version
                # if this doesn't work, you could try the following although I did not verify this works:
                echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> ~/.zshrc
                source ~/.zshrc
                julia --version

                # Then in Julia:
                using Pkg
                Pkg.activate(".")  # optional but recommended: use this repo as the active Julia project
                Pkg.develop(path="dependencies/EPEDNN.jl")
                Pkg.instantiate()

                #Then in Julia: 
                using EPEDNN


                Make sure that EPEDNN submodule is installed, the General Julia package registry is installed, and juliacall and juliapkg are installed.
                The following commands may help:

                pip install juliapkg
                git submodule update --init dependencies/EPEDNN.jl
                # Only needed if juliapkg fails on registry download:
                git clone --depth 1 https://github.com/JuliaRegistries/General.git \
                ~/.julia/registries/General
                pip install juliacall
            '''
            # you can run Julia commands in Python using jl.seval('command')
            # jl.seval('using Pkg')
            # jl.seval('Pkg.activate(".")  # optional but recommended: use this repo as the active Julia project')
            # jl.seval('Pkg.develop(path="/Users/nelsonlab/codes/saarelma-conner-ped/dependencies/EPEDNN.jl")')
            # jl.seval('Pkg.instantiate()')
            # jl.seval('using EPEDNN')

            # 2. Load the pre-trained EPED neural network model
            # This mimics the EPEDNN.loadmodelonce("EPED1NNmodel.bson") step
            model_filename = "EPED1NNmodel.bson" 
            self.epednn_model = jl.EPEDNN.loadmodelonce(model_filename)
        elif model in ('EPED_SPARC', 'EPED_SCOPING'):
            # Vendored package lives at dependencies/epednn_mit/src/epednn_mit/;
            # it is not installed into the env by default, so put src/ on sys.path.
            import sys
            import importlib
            from pathlib import Path
            epednn_root = Path(__file__).resolve().parents[2] / "dependencies" / "epednn_mit" # ESCAPE root (src/epednn/..)
            epednn_src = epednn_root / "src"
            if not epednn_src.is_dir():
                raise FileNotFoundError(
                    f"epednn_mit not found at {epednn_src}. "
                    "Expected dependencies/epednn_mit/src in the repo."
                )
            if str(epednn_src) not in sys.path:
                sys.path.insert(0, str(epednn_src))
            # The two model families share a layout, so select the subpackage by name.
            subpkg = {'EPED_SPARC': 'sparc', 'EPED_SCOPING': 'scoping'}[model]
            weights_dir = epednn_src / "epednn_mit" / "models" / subpkg
            if not weights_dir.is_dir():
                raise FileNotFoundError(
                    f"epednn_mit model '{subpkg}' not found at {weights_dir}. "
                    "The scoping model lives on the 'scoping_model' branch of the submodule: "
                    "git -C dependencies/epednn_mit checkout scoping_model"
                )
            module = importlib.import_module(f"epednn_mit.models.{subpkg}.tensorflow_model")
            generate = getattr(module, f"generate_epednn_mit_{subpkg}_tensorflow")
            from epednn_mit.utils.load import load_weights
            weights = load_weights(sorted(weights_dir.glob(f"*{subpkg}*.pkl")))
            if not weights:
                raise FileNotFoundError(f"No *{subpkg}*.pkl weights found in {weights_dir}")
            self.epednn_model = generate(weights)

        self.bt = np.array(self.calc_B(self.eq['raxis'],self.eq['zaxis'])[1][2])

    def feed_epednn(self, model='EPED1', pres_gfile=False, ne_ped_h=None, psin_ped=None):
        """Run the EPEDNN model
        
        parameters
        ____________________________________________
        ne_ped_h : float
        specify if you want to manually specify the pedestal height (likely only for testing)

        psin_ped : float
        specify if you want to manually specify the pedestal height using the pedestal width
        """
        
        # Define inputs in Python
        if ne_ped_h is not None:
            self.ne_ped_h = ne_ped_h
        elif psin_ped is not None:
            self.ne_ped_h = interp1d(self.psi_ne_eval,self.n_e, kind='linear', bounds_error=False, fill_value='extrapolate')(1-psin_ped)
        else:
            raise NotImplementedError('need to specify either a radial location of or the actual value of the density to give to EPEDNN as ne_ped')
        
        self.calc_betan(pres_gfile)
        print(f'betan: {self.betan}')

        inputs = {
            "a": float(self.a),           # Minor radius (m)
            "betan": float(self.betan[0]),       # Normalized beta
            "bt": float(abs(self.bt[0])),                # Toroidal magnetic field at the magnetic axis (T)
            "delta": float(self.delta),       # Effective triangularity
            "ip": float(abs(self.Ip)),          # Plasma current (MA)
            "kappa": float(self.kappa),       # Elongation
            "m": float(self.M_eff),           # Effective mass (must be 2.0 for D or 2.5 for D-T)
            "neped": float(self.ne_ped_h / (1e19)),       # Pedestal density (in 10^19 m^-3)
            "r": float(self.Rmajor),           # Major radius (m)
            "zeffped": float(self.Z_eff)      # Effective charge
        }

        if model == 'EPED1':
            # Call the Julia model using the Python inputs
            # We pass the inputs into the Julia function, along with the keyword arguments
            solution = self.epednn_model(
                inputs["a"], 
                inputs["betan"], 
                inputs["bt"], 
                inputs["delta"],
                inputs["ip"], 
                inputs["kappa"], 
                inputs["m"], 
                inputs["neped"],
                inputs["r"], 
                inputs["zeffped"],
                only_powerlaw=False,        # Set to True if you only want the scaling law
                warn_nn_train_bounds=True   # Warns if inputs are outside the training data. Good for debugging
            )

            # Extract the results back into Python
            # The solution structure has pressure and width for different modes (GH, G, H)
            self.pedestal_pressure = solution.pressure.GH.H  # in MPa
            self.pedestal_width = solution.width.GH.H        # in normalized poloidal flux

        elif model == 'EPED_SPARC':
            ''' Training dataset was on (in order of input position from the EPEDNN_MIT README): 
            Ip:     [  1.6  , 14.3   ]
            Bt:     [  7.2  , 12.2   ]
            R:      [  1.85 ,  1.85  ]
            a:      [  0.57 ,  0.57  ]
            kappa:  [  1.53 ,  2.29  ]
            delta:  [  0.39 ,  0.59  ]
            neped:  [  2.84 , 90.235 ]
            betan:  [  0.8  ,  1.6   ]
            zeff:   [  1.3  ,  2.5   ]
            '''
            if (inputs["bt"] - 12.2 < 0.5) and (inputs["bt"] > 12.2):
                print('Warning: bt is close to 12.2 but is greater than 12.2, setting bt to 12.2')
                inputs["bt"] = 12.2
            if (inputs["a"] - 0.57 < 1e-3) and (inputs["a"] != 0.57):
                print('Warning: a is close to 0.57, setting a to 0.57 or else EPEDNN behaves badly')
                inputs["a"] = 0.57
            if (inputs["r"] - 1.85 < 1e-3) and (inputs["r"] != 1.85):
                print('Warning: r is close to 1.85, setting r to 1.85 or else EPEDNN behaves badly')
                inputs["r"] = 1.85
            
            x = np.atleast_2d([
                inputs["ip"], 
                inputs["bt"], 
                inputs["r"], 
                inputs["a"], 
                inputs["kappa"], 
                inputs["delta"], 
                inputs["neped"], 
                inputs["betan"], 
                inputs["zeffped"]
            ])
            solution = self.epednn_model.predict(x)[0]  # [[ped_height, ped_width]]
            print(solution)
            self.pedestal_pressure = solution[0] / 1000     # in MPa -> kPa
            self.pedestal_width = solution[1]              # in normalized poloidal flux

        elif model == 'EPED_SCOPING':
            '''The minimum and maximum of the training dataset input ranges used to generate these models are below, in order of input position:

            a:      [  0.4  ,   2.2  ]
            aspect: [  2.0  ,   4.2  ]
            kappa:  [  1.3  ,   2.5  ]
            delta:  [  0.3  ,   0.7  ]
            bt:     [  2.0  ,  18.0  ]  * Not a clean boundary so (3.0, 17.0) might be more prudent
            qstar:  [  3.0  ,   5.0  ]
            betan:  [  0.3  ,   3.7  ]
            zeff:   [  1.2  ,   3.2  ]
            fgped:  [  0.3  ,   1.3  ]
            nsfrac: [  0.2  ,   0.8  ]
            tesep:  [ 50.0  , 500.0  ]'''

            inputs['aspect'] = inputs['r'] / inputs['a']
            inputs['tesep'] = self.T_e[-1] * 1000 # eV
            n_GW = inputs['ip'] / (np.pi * inputs['a']**2)  # m^-3
            inputs['fgped'] = inputs['neped'] * 1e-1 / n_GW

            x = np.atleast_2d([
                inputs["a"], 
                inputs["aspect"],
                inputs["kappa"],
                inputs["delta"],
                inputs["bt"], 
                inputs["qstar"], # missing
                inputs["betan"],
                inputs["zeffped"],
                inputs["fgped"],
                inputs["nsfrac"], # missing
                inputs["tesep"]
            ])
            solution = self.epednn_model.predict(x)[0]  # [[ped_height, ped_width]]
            print(solution)
            self.pedestal_pressure = solution[0] / 1000     # in MPa -> kPa
            self.pedestal_width = solution[1]              # in normalized poloidal flux


        return self.pedestal_pressure, self.pedestal_width, self.betan
