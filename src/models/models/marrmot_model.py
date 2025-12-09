import numpy as np
import numbers

from scipy.optimize import fsolve, least_squares

import pymarrmot.functions.objective_functions as objective_functions
from pymarrmot.functions.solver_functions import NewtonRaphson as nr

import cma as cma

class MARRMoT_model:
    """
    Superclass for all MARRMoT models
    """
    
    def __init__(self):

        #static attributes, set for each model in the model definition
        self.num_stores = None           #number of model stores
        self.num_fluxes = None           #number of model fluxes
        self.num_params = None           #number of model parameters
        self.par_ranges = None           #default parameter ranges
        self.jacob_pattern = None        #pattern of the Jacobian matrix of model store ODEs
        self.store_names = None          #names for the stores
        self.flux_names = None           #Names for the fluxes
        self.flux_groups = None          #Grouping of fluxes (useful for water balance and output)
        self.StoreSigns = None          #Signs to give to stores (-1 is a deficit store), assumes all 1 if not given
                                        
        #attributes set at the beginning of the simulation directly by the user
        self.theta = None               #Set of parameters
        self.delta_t = None             #time step
        self.S0 = None                  #initial store values
        self.input_climate = None       #vector of input climate
        self.solver_opts = None         #options for numerical solving of ODEs
                                        #automatically, based on parameter set
        self.store_min = None           #store minimum values
        self.store_max = None           #store maximum values

        #attributes created and updated automatically throughout a simulation
        self.t = None                   #current timestep
        self.fluxes = None              #vector of all fluxes
        self.stores = None              #vector of all stores
        self.uhs = None                 #unit hydrographs and still-to-flow fluxes
        self.solver_data = None         #step-by-step info of solver used and residuals
        self.status = None              #0 = model created, 1 = simulation ended
    
    def __setattr__(self, name, value):
        """
        Set methods with checks on inputs for attributes set by the user
        """
        if name == 'theta':
            if value is not None:
                if value.size == self.num_params:
                    super().__setattr__(name, value)
                    self.reset()
                else:
                    raise ValueError(f'theta must have {self.num_params} elements')
            else:
                super().__setattr__(name, value)   
        elif name == 'delta_t':
            if value is not None:
                if isinstance(value, numbers.Number):
                    super().__setattr__(name, value)
                    self.reset()
                else:
                    raise ValueError('delta_t must be a scalar')
            else:
                super().__setattr__(name, value)       
        elif name == 'input_climate':
            if value is not None:
                if isinstance(value, dict):
                    if all(key in value for key in ('precip', 'pet', 'temp')):
                        value['precip'] = value['precip'] / self.delta_t
                        value['pet'] = value['pet'] / self.delta_t
                        
                        # ERROR? Why divide temperature by delta-t?
                        value['temp'] = value['temp'] / self.delta_t
                        
                        super().__setattr__(name, value)
                        self.reset()
                    else:
                        raise ValueError('Input climate dictionary must contain fields: precip, pet, temp')
            else:
                super().__setattr__(name, value)
        elif name == 'S0':
            if value is not None:
                if value.size == self.num_stores:
                    super().__setattr__(name, value)
                    self.reset()
                else:
                    raise ValueError(f'S0 must have {self.num_stores} elements')
            else:
                super().__setattr__(name, value)
        elif name == 'solver_opts':
            if value is not None:
                if isinstance(value, dict):
                    super().__setattr__(name, self.add_to_def_opts(value))
                    self.reset()
                else:
                    raise ValueError('solver_opts must be a dictionary')
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def reset(self):
        """
        RESET is called any time that a user-specified input is changed 
        (t, delta_t, input_climate, S0, solver_options) and resets any
        previous simulation ran on the object.
        This is to prevent human error in analysing results.
        """
        self.t = None             # current timestep
        self.fluxes = None        # vector of all fluxes
        self.stores = None        # vector of all stores
        self.uhs = None           # unit hydrographs and still-to-flow fluxes
        self.solver_data = None   # step-by-step info of solver used and residuals
        self.status = 0           # 0 = model created, 1 = simulation ended

    def init_(self):
        """
        INIT_ runs before each model run to initialise store limits,
        auxiliary parameters etc. it calls INIT which is model specific
        """
        self.store_min = np.zeros((self.num_stores, 1))
        self.store_max = np.inf * np.ones((self.num_stores, 1))
        t_end = self.input_climate['precip'].shape[0]
        self.stores = np.zeros((t_end, self.num_stores))
        self.fluxes = np.zeros((t_end, self.num_fluxes))
        self.solver_data = {
            'resnorm': np.zeros(t_end),
            'solver': np.full(t_end, 'None'),
            #'solver': np.zeros(t_end),
            'iter': np.zeros(t_end)
        }
        self.init()

    def ODE_approx_IE(self, S):
        """
        ODE approximation with Implicit Euler time-stepping scheme.

        Parameters:
        -----------
        S : numpy.ndarray
            State variables.

        Returns:
        --------
        err : numpy.ndarray
            Error in the approximation.
        """
        S = S.ravel()
        # model_fun returns the ODEs (changes in storage) and fluxes
        result = self.model_fun(S)
        delta_S = result[0]
        if self.t == 0:
            Sold = self.S0.ravel()
        else:
            Sold = self.stores[self.t - 1, :].ravel()
        err = (S - Sold) / self.delta_t - delta_S
        return err

    def solve_stores(self, s_old):
        """
        SOLVE_STORES solves the stores ODEs
        
        Parameters:
        -----------
        self : object
            Instance of the class containing solver options.
        Sold : array_like
            Array of shape (num_stores,) containing the initial storage values.

        Returns:
        --------
        tuple
            A tuple containing the following:
                Snew : ndarray
                    Array of shape (num_stores,) containing the new storage values.
                resnorm : float
                    The residual norm of the solution.
                solver : str
                    The name of the solver used ('NewtonRaphson', 'fsolve', or 'lsqnonlin').
                iter : int
                    Number of iterations taken by the solver.
        """
        solver_opts = self.solver_opts

        # Reduce tolerance to a fraction of the smallest store
        resnorm_tolerance = solver_opts['resnorm_tolerance'] * min(abs(s_old)) + 1E-5
        # Create vectors for each solver
        Snew_v = np.zeros((3, self.num_stores))
        resnorm_v = np.full(3, np.inf)
        iter_v = np.ones(3, dtype=int)

        # NewtonRaphson solver
        tmp_Snew, tmp_fval, exit_flag = nr.NewtonRaphson(self.ODE_approx_IE, s_old, solver_opts['NewtonRaphson'])
       
        if(tmp_Snew<self.store_min.flatten()).any() or (tmp_Snew > self.store_max.flatten()).any():
            tmp_Snew = np.max([tmp_Snew, self.store_min.flatten()], axis=0)
            tmp_Snew = np.min([tmp_Snew, self.store_max.flatten()], axis=0)
            tmp_fval = self.ODE_approx_IE(tmp_Snew)
        
        tmp_resnorm = np.sum(tmp_fval**2)
        Snew_v[0, :], resnorm_v[0] = tmp_Snew, tmp_resnorm

        # If NewtonRaphson doesn't find a good enough solution, run fsolve
        if tmp_resnorm > resnorm_tolerance:
            tmp_Snew, tmp_fval, _, tmp_iter = self.rerunSolver('fsolve', tmp_Snew, s_old)
            tmp_resnorm = np.sum(tmp_fval**2)

            Snew_v[1, :], resnorm_v[1], iter_v[1] = tmp_Snew, tmp_resnorm, tmp_iter

            # If fsolve doesn't find a good enough solution, run lsqnonlin
            if tmp_resnorm > resnorm_tolerance:
                tmp_Snew, tmp_fval, _, tmp_iter = self.rerunSolver('lsqnonlin', tmp_Snew, s_old)
                tmp_resnorm = np.sum(tmp_fval**2)

                Snew_v[2, :], resnorm_v[2], iter_v[2] = tmp_Snew, tmp_resnorm, tmp_iter

        # Get the best solution
        best_solver_id = np.argmin(resnorm_v)
        Snew, resnorm, iter = Snew_v[best_solver_id], resnorm_v[best_solver_id], iter_v[best_solver_id]

        solver = ["NewtonRaphson", "fsolve", "lsqnonlin"][best_solver_id]

        return Snew, resnorm, solver, iter

    def rerunSolver(obj, solverName, initGuess, Sold):
        '''
        rerunSolver - Restarts a root-finding solver with different starting points.
        
        Parameters:
        -----------
        obj : object
            Instance of the class containing solver options.
        solverName : str
            Name of the solver to be used ('fsolve' or 'lsqnonlin').
        initGuess : array_like
            Initial guess for the solution.
        Sold : array_like
            Array of shape (num_stores,) containing the previous storage values.

        Returns:
        --------
        tuple
            A tuple containing the following:
                Snew : ndarray
                    Array of shape (num_stores,) containing the new storage values.
                fval : ndarray
                    The value of the function at the solution.
                stopflag : int
                    Flag indicating the reason for stopping the solver (0 for iteration count exceeded, 1 for normal function run).
                stopiter : int
                    Number of iterations taken by the solver.
        '''
        solver_opts = obj.solver_opts[solverName]
        solve_fun = obj.ODE_approx_IE
        max_iter = obj.solver_opts['resnorm_maxiter']
        resnorm_tolerance = obj.solver_opts['resnorm_tolerance'] * min(abs(Sold)) + 1E-5

        iter_count = 1
        resnorm = resnorm_tolerance + 1
        num_stores = obj.num_stores
        stopflag = 1
        Snew = -np.ones(num_stores)
        
        Snew_v = np.zeros((num_stores, max_iter))
        fval_v = np.full((num_stores, max_iter), np.inf)
        resnorm_v = np.full(max_iter, np.inf)

        while resnorm > resnorm_tolerance:
            if iter_count == 1:
                x0 = initGuess
            elif iter_count == 2:
                x0 = Sold
            elif iter_count == 3:
                x0 = np.maximum(-2 * 10**4, obj.store_min.flatten())
            elif iter_count == 4:
                x0 = np.minimum(2 * 10**4, obj.store_max.flatten())
            else:
                x0 = np.maximum(0, Sold + np.random.randn(num_stores) * Sold / 10)

            #print(f'{iter_count}.x0={x0}')
            if solverName.lower() == 'fsolve':
                Snew_tmp, info_dict, ier, mesg = fsolve(solve_fun, x0, full_output = True)
                fval_v[:,iter_count-1] = info_dict['fvec']
                if(Snew_tmp<obj.store_min.flatten()).any() or (Snew_tmp > obj.store_max.flatten()).any():
                    Snew_tmp = np.max([Snew_tmp, obj.store_min.flatten()], axis=0)
                    Snew_tmp = np.min([Snew_tmp, obj.store_max.flatten()], axis=0)
                    fval_v[:, iter_count - 1] = solve_fun(Snew_tmp)
                Snew_v[:,iter_count-1] = Snew_tmp
               
            elif solverName.lower() == 'lsqnonlin':
                solver_opts2 = {}
                solver_opts2['max_nfev'] = solver_opts['MaxFunEvals']
                if(x0<obj.store_min.flatten()).any() or (x0 > obj.store_max.flatten()).any():
                    x0 = np.max([x0, obj.store_min.flatten()], axis=0)
                    x0 = np.min([x0, obj.store_max.flatten()], axis=0)
                res = least_squares(solve_fun, x0, bounds=(obj.store_min.flatten(), obj.store_max.flatten()), method='trf', **solver_opts2)
                #print(f'!!!Least Squares. Result = {res.x}')
                Snew_v[:, iter_count - 1] = res.x
                fval_v[:, iter_count - 1] = res.fun
                stopflag = res.status
            else:
                raise ValueError("Only fsolve and lsqnonlin are supported.")

            resnorm_v[iter_count - 1] = np.sum(fval_v[:, iter_count - 1]**2)
            min_resnorm_idx = np.argmin(resnorm_v)
            resnorm = resnorm_v[min_resnorm_idx]
            fval = fval_v[:, min_resnorm_idx]
            Snew = Snew_v[:, min_resnorm_idx]

            if iter_count >= max_iter:
                ##Stopflag = 5 indicates that rerun solver iterations exceeds the max_iter criteria
                ##By using 5 we reserve the -1 to 4 values returned by the least squares solver
                stopflag = 5
                break

            iter_count += 1

            
            

        return Snew, fval, stopflag, iter_count

    def rerunSolver_MatLab(obj, solverName, initGuess, Sold):
        """
        rerunSolver - Restarts a root-finding solver with different starting points.
        
        Parameters:
        -----------
        obj : object
            Instance of the class containing solver options.
        solverName : str
            Name of the solver to be used ('fsolve' or 'lsqnonlin').
        initGuess : array_like
            Initial guess for the solution.
        Sold : array_like
            Array of shape (num_stores,) containing the previous storage values.

        Returns:
        --------
        tuple
            A tuple containing the following:
                Snew : ndarray
                    Array of shape (num_stores,) containing the new storage values.
                fval : ndarray
                    The value of the function at the solution.
                stopflag : int
                    Flag indicating the reason for stopping the solver (0 for iteration count exceeded, 1 for normal function run).
                stopiter : int
                    Number of iterations taken by the solver.
        """
        solver_opts = obj.solver_opts[solverName]
        solve_fun = obj.ODE_approx_IE
        max_iter = obj.solver_opts['resnorm_maxiter']
        resnorm_tolerance = obj.solver_opts['resnorm_tolerance'] * min(abs(Sold)) + 1E-5

        iter_count = 1
        resnorm = resnorm_tolerance + 1
        num_stores = obj.num_stores
        stopflag = 1
        Snew = -np.ones(num_stores)
        
        Snew_v = np.zeros((num_stores, max_iter))
        fval_v = np.full((num_stores, max_iter), np.inf)
        resnorm_v = np.full(max_iter, np.inf)

        while resnorm > resnorm_tolerance:
            if iter_count == 1:
                x0 = initGuess
            elif iter_count == 2:
                x0 = Sold
            elif iter_count == 3:
                x0 = np.maximum(-2 * 10**4, obj.store_min)
            elif iter_count == 4:
                x0 = np.minimum(2 * 10**4, obj.store_max)
            else:
                x0 = np.maximum(0, Sold + np.random.randn(num_stores) * Sold / 10)

            if solverName.lower() == 'fsolve':
                Snew_v, info_dict = fsolve(solve_fun, x0)
                ##Snew_v[:, iter_count - 1], fval_v[:, iter_count - 1], _, stopflag = fsolve(solve_fun, x0)
            elif solverName.lower() == 'lsqnonlin':
                res = least_squares(solve_fun, x0, bounds=(obj.store_min, np.inf), method='trf', **solver_opts)
                Snew_v[:, iter_count - 1] = res.x
                fval_v[:, iter_count - 1] = res.fun
                stopflag = res.status
            else:
                raise ValueError("Only fsolve and lsqnonlin are supported.")

            resnorm_v[iter_count - 1] = np.sum(fval_v[:, iter_count - 1]**2)
            min_resnorm_idx = np.argmin(resnorm_v)
            resnorm = resnorm_v[min_resnorm_idx]
            fval = fval_v[:, min_resnorm_idx]
            Snew = Snew_v[:, min_resnorm_idx]

            if iter_count >= max_iter:
                stopflag = 0
                break

            iter_count += 1

        return Snew, fval, stopflag, iter_count
    
    def run(obj, input_climate=None, S0=None, theta=None, solver_opts=None):
        """
        RUN runs the model with a given climate input, initial stores, parameter set, and solver settings.

        Parameters:
        -----------
        input_climate : array_like, optional
            Climate input data. Default is None.
        S0 : array_like, optional
            Initial stores. Default is None.
        theta : array_like, optional
            Parameter set. Default is None.
        solver_opts : dict, optional
            Solver settings. Default is None.

        Returns:
        --------
        None
        """
        if solver_opts is not None:
            obj.solver_opts = solver_opts
        if theta is not None:
            obj.theta = theta
        if S0 is not None:
            obj.S0 = S0
        if input_climate is not None:
            obj.input_climate = input_climate

        obj.init_()

        t_end = obj.input_climate['precip'].shape[0]

        for t in range(t_end):
            obj.t = t
            if t == 0:
                Sold = obj.S0.ravel()
            else:
                Sold = obj.stores[t - 1, :].ravel()

            Snew, resnorm, solver, iter = obj.solve_stores(Sold)

            dS, f = obj.model_fun(Snew)

            obj.fluxes[t, :] = f * obj.delta_t

            obj.stores[t, :] = Sold + dS * obj.delta_t

            obj.solver_data['resnorm'][t] = resnorm
            obj.solver_data['solver'][t] = solver
            obj.solver_data['iter'][t] = iter

            obj.step()

        obj.status = 1

    def get_output(obj, nargout, *args):
        """
        GET_OUTPUT runs the model exactly like RUN, but output is consistent with current MARRMoT.

        Parameters:
        -----------
        nargout : int
            number of output arguments.
        *args : array_like, optional
            Additional arguments.
            

        Returns:
        --------
        fluxOutput : dict
            Fluxes leaving the model.
        fluxInternal : dict
            Fluxes internal to the model.
        storeInternal : dict
            Stores internal to the model.
        waterBalance : dict, optional
            Water balance data if requested.
        solverSteps : dict, optional
            Step-by-step data of the solver if requested.
        """
        if args or obj.status is None or obj.status == 0:
            obj.run(*args)

        fluxOutput = {}
        for fg in obj.flux_groups:
            idx = np.abs(obj.flux_groups[fg])
            signs = np.sign(obj.flux_groups[fg])
            #indices for fluxes are 1-based in model definition files, so convert to 0-based
            idx = idx - 1
            
            #bug fix: 08Aug2024 - SAS - original code creates array signs, which seems to be unnecessary,
            #as idx is the absolute value of the flux group indices
            #the code is therefore rearranged to eliminate signs from the calculation
            #original code (can result in sign = 0): signs = np.sign(obj.flux_groups[fg]) (see Matlab code for use
            #of signs in calculation of fluxOutput)
            if idx.size == 1:
                fluxOutput[fg] =  signs * obj.fluxes[:, idx]
            else:
                fluxOutput[fg] = np.sum(signs * obj.fluxes[:, idx], axis=1)

        fluxInternal = {obj.flux_names[i]: obj.fluxes[:, i] for i in range(obj.num_fluxes)}

        storeInternal = {obj.store_names[i]: obj.stores[:, i] for i in range(obj.num_stores)}

        if nargout == 4:
            waterBalance = obj.check_waterbalance()
            return fluxOutput, fluxInternal, storeInternal, waterBalance
        elif nargout == 5:
            waterBalance = obj.check_waterbalance()
            solverSteps = obj.solver_data
            return fluxOutput, fluxInternal, storeInternal, waterBalance, solverSteps
        else:
            return fluxOutput, fluxInternal, storeInternal

    def check_waterbalance(obj, *args):
        """
        CHECK_WATERBALANCE returns the water balance.

        Parameters:
        -----------
        *args : array_like, optional
            Additional arguments.

        Returns:
        --------
        out : float
            Water balance value.
        """
        if args or not obj.status or obj.status == 0:
            obj.run(*args)

        P = obj.input_climate['precip'][()]
        fg = obj.flux_groups.keys()
        OutFluxes = np.zeros(len(fg))
        for k in range(len(fg)):
            idx = np.abs(obj.flux_groups[list(fg)[k]])
            signs = np.sign(obj.flux_groups[list(fg)[k]])
            OutFluxes[k] = np.sum(signs*obj.fluxes[:,idx-1])
            # OpenAI interpretation: OutFluxes[k] = np.sum(np.sum(signs * obj.fluxes[:, idx], 1), 2)
        
        if obj.StoreSigns is None:
            obj.StoreSigns = np.ones(obj.num_stores)
        dS = obj.StoreSigns * (obj.stores[-1] - obj.S0)
        if obj.uhs is None:
            R = np.array([])
        else:
            R = np.sum([np.sum(uh[1]) for uh in obj.uhs], axis=0)

        out = np.sum(P) - np.sum(OutFluxes) - np.sum(dS) - np.sum(R)

        """ This piece was relevant in Matlab, but not in Python
        print('Total P  =', np.sum(P), 'mm.')
        for k, fg_k in enumerate(fg):
            print('Total', fg_k, '=', -OutFluxes[k], 'mm.')
        for s in range(obj.num_stores):
            if obj.StoreSigns[s] == -1:
                ending = ' (deficit store).'
            else:
                ending = '.'
            print('Delta S{} ='.format(s + 1), -dS[s], 'mm', ending)
        if R.size != 0:
            print('On route =', -np.sum(R), 'mm.')

        print('-------------')
        print('Water balance =', out, 'mm.') """
        
        return out

    def get_streamflow(obj, pars):
        """
        GET_STREAMFLOW only returns the streamflow, runs the model if it hadn't run already.

        Parameters:
        -----------
        *args : array_like, optional
            Additional arguments.

        Returns:
        --------
        Q : array_like
            Streamflow values.
        """
        if pars.all() or not obj.status or obj.status == 0:
            obj.run(theta=pars)

        q = [x - 1 for x in obj.flux_groups['Q']]
        Q = np.sum(obj.fluxes[:,q], axis=1)
        #Q = np.sum(obj.fluxes[:, obj.flux_groups['Q']], axis=1)
        return Q
        
    def calibrate(obj, Q_obs, cal_idx, optim_fun, par_ini=None, optim_opts=None, of_name=None,
              inverse_flag=None, display=None, *args):
        """
        CALIBRATE uses the chosen algorithm to find the optimal parameter set, given model inputs, objective function,
        and observed streamflow.

        Parameters:
        -----------
        obj : object
            The instance of the model.
        Q_obs : array_like
            Observed streamflow.
        cal_idx : array_like
            Timesteps to use for model calibration.
        optim_fun : function
            Function to use for optimization.
        par_ini : array_like, optional
            Initial parameter estimates. Default is None.
        optim_opts : dict, optional
            Options for the optimization function. Default is None.
        of_name : function, optional
            Name of the objective function to use. Default is None.
        inverse_flag : int, optional
            If the objective function should be inversed. Default is None.
        display : bool, optional
            If information about the calibration should be displayed. Default is None.
        *args : array_like
            Additional arguments to the objective function.

        Returns:
        --------
        par_opt : array_like
            Optimal parameter set.
        of_cal : float
            Value of the objective function at par_opt.
        stopflag : int
            Flag indicating the reason the algorithm stopped.
        output : dict
            Output, see the documentation of the optimization function for detail.
        """

        
        
        if (obj.input_climate is None) or (obj.delta_t is None) or (obj.S0 is None) or (obj.solver_opts is None):
            raise ValueError('input_climate, delta_t, S0, and solver_opts attributes must be specified before calling calibrate.')

        # if the list of timesteps to use for calibration is empty, use all timesteps
        if cal_idx is None:
            cal_idx = np.arange(1,len(Q_obs)+1)

        if display is None:
            display = True

        if isinstance(cal_idx, np.ndarray):
            cal_idx = cal_idx.tolist()
        if isinstance(cal_idx, list):
            cal_idx = sorted(cal_idx)

        if display:
            print('---')
            print(f'Starting calibration of model {type(obj)}.')
            print(f'Simulation will run for timesteps 1-{max(cal_idx)}.')
            print(f'Objective function {of_name} will be calculated in time steps {cal_idx[0]} through {cal_idx[len(cal_idx)-1]}.')
            print(f'The optimiser {optim_fun} will be used to optimise the objective function.')
            print('Options passed to the optimiser:')
            print(optim_opts)
            print('All other options are left to their default, check the source code of the optimiser to find these default values.')
            print('---')

        input_climate_all = obj.input_climate.copy()

        # Use the data from the start to the last value of cal_idx to run the model
        obj.input_climate.clear()
        obj.input_climate['dates'] = input_climate_all['dates'][0:max(cal_idx)]
        obj.input_climate['precip'] = input_climate_all['precip'][0:max(cal_idx)]
        obj.input_climate['temp'] = input_climate_all['temp'][0:max(cal_idx)]
        obj.input_climate['pet'] = input_climate_all['pet'][0:max(cal_idx)]
        Q_obs = Q_obs[:max(cal_idx)]

        if par_ini is None:
            par_ini = np.mean(obj.par_ranges, axis=1)

        # Helper function to calculate fitness given a set of parameters
        def fitness_fun(par):
            print(par)
            Q_sim = obj.get_streamflow(par)
            obj_func = getattr(objective_functions, of_name)
            result = obj_func(Q_obs, Q_sim, cal_idx, *args)
            if inverse_flag:
                fitness = (-1) * result[0]
            else:
                fitness = result[0]    
            return fitness

        #Documentation states, to provide sigma values for each parameter, specify sigma = 1, then CMA_stds as multipliers for each parameter
        xopt, es = cma.fmin2(fitness_fun, par_ini, 1, {'CMA_stds': optim_opts['insigma'], 'bounds': [optim_opts['LBounds'], optim_opts['UBounds']], 'seed': 13})

        obj.input_climate.clear()
        obj.input_climate = input_climate_all.copy()

        return es.best.x, es.best.f, es.stop(), es.result
    
    def default_solver_opts(obj):
        """
        Function to return default solver options.

        Returns:
        --------
        solver_opts : dict
            Dictionary containing default solver options.
        """
        solver_opts = {
            'resnorm_tolerance': 0.1,  # Root-finding convergence tolerance
            'resnorm_maxiter': 6,      # Maximum number of re-runs used in rerunSolver
            'NewtonRaphson': {'MaxIter': obj.num_stores * 10},
            'fsolve': {'jacob_pattern': obj.jacob_pattern},
            'lsqnonlin': {'jacob_pattern': obj.jacob_pattern, 'MaxFunEvals': 1000}
        }

        # In the original code there were two separate
        # if not obj.isOctave:
        #     solver_opts['fsolve'] = {'Display': 'none', 'jacob_pattern': obj.jacob_pattern}
        #     solver_opts['lsqnonlin'] = {'Display': 'none', 'jacob_pattern': obj.jacob_pattern, 'MaxFunEvals': 1000}
        # else:
        #     solver_opts['fsolve'] = {'Display': 'off'}
        #     solver_opts['lsqnonlin'] = {'Display': 'off', 'MaxFunEvals': 1000}

        return solver_opts

    def add_to_def_opts(self, opts=None):
        """
        Function to add new solver options to the default ones.

        Parameters:
        -----------
        opts : dict, optional
            New solver options to add. Default is None.

        Returns:
        --------
        solver_opts : dict
            Dictionary containing the merged solver options.
        """
        def_opts = self.default_solver_opts()
        if opts is None:
            solver_opts = def_opts
        else:
            solver_opts = {}
            for field, default_value in def_opts.items():
                if field not in opts or opts[field] is None:
                    solver_opts[field] = default_value
                elif isinstance(default_value, dict):
                    solver_opts[field] = {**default_value, **opts[field]}
                else:
                    solver_opts[field] = opts[field]
        return solver_opts

    
    
    
    
    
    
    
    
'''
Not sure if this stuff is used, but not ready to get rid of it yet...
    def solve_fluxes(self, Sold):
        fluxes = np.zeros((self.num_fluxes, 1))
        fluxes[self.uhs] = Sold[self.uhs] / self.theta[self.uhs]
        return fluxes

    def integrate_store(self, Sold):
        self.stores[self.t] = Sold.reshape(-1)
        self.fluxes[self.t] = self.solve_fluxes(Sold).reshape(-1)

    def integrate_one_step(self):
        Sold = self.stores[self.t - 1].reshape(-1, 1)
        Snew, resnorm, iter_ = self.solve_stores(Sold)
        self.solver_data['resnorm'][self.t] = resnorm
        self.solver_data['iter'][self.t] = iter_
        self.integrate_store(Snew)

    def integrate(self):
        self.init_()
        for self.t in range(1, self.input_climate.shape[0]):
            self.integrate_one_step()

    def get_fluxes(self, theta=None):
        if theta is not None:
            self.theta = theta
        if self.t is None:
            self.integrate()
        return self.fluxes

    def get_stores(self, theta=None):
        if theta is not None:
            self.theta = theta
        if self.t is None:
            self.integrate()
        return self.stores

    def get_sim_data(self, theta=None):
        if theta is not None:
            self.theta = theta
        if self.t is None:
            self.integrate()
        return self.stores, self.fluxes

    def add_to_def_opts(self, new_opts):
        default_opts = {
            'resnorm_tolerance': 1e-5,
            'resnorm_maxiter': 1000
        }
        default_opts.update(new_opts)
        return default_opts

    def check_input(self):
        assert self.num_stores == len(self.store_names), 'num_stores must equal length of store_names'
        assert self.num_fluxes == len(self.flux_names), 'num_fluxes must equal length of flux_names'
        assert self.num_params == len(self.par_ranges), 'num_params must equal length of par_ranges'
        assert self.num_params == len(self.jacob_pattern), 'num_params must equal length of jacob_pattern'

    def jacob_fun(self, S):
        S = S.reshape(-1, 1)
        dfdS = self.model_jacob(S)
        return csr_matrix(dfdS)

    def solve_stores_nl(self, Sold):
        solver_opts = self.solver_opts
        resnorm_tolerance = solver_opts['resnorm_tolerance']
        resnorm_maxiter = solver_opts['resnorm_maxiter']

        resnorm_fun = lambda S: np.sum(self.ODE_approx_IE(S)**2)
        dfdS_fun = lambda S: self.jacob_fun(S)

        Snew, info = lsqnonlin(resnorm_fun, Sold, jac=dfdS_fun, bounds=(self.store_min.flatten(), self.store_max.flatten()), ftol=resnorm_tolerance, max_nfev=resnorm_maxiter, full_output=True)[:2]

        resnorm = np.sum(info['fvec']**2)
        iter_ = info['nfev']

        return Snew.reshape(-1, 1), resnorm, iter_

    def integrate_one_step_nl(self):
        Sold = self.stores[self.t - 1].reshape(-1, 1)
        Snew, resnorm, iter_ = self.solve_stores_nl(Sold)
        self.solver_data['resnorm'][self.t] = resnorm
        self.solver_data['iter'][self.t] = iter_
        self.integrate_store(Snew)

    def integrate_nl(self):
        self.init_()
        for self.t in range(1, self.input_climate.shape[0]):
            self.integrate_one_step_nl()

    def get_sim_data_nl(self):
        if self.t is None:
            self.integrate_nl()
        return self.stores, self.fluxes

    def model_fun(self, S):
        raise NotImplementedError

    def model_jacob(self, S):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError
'''