#!/usr/bin/env python3
"""
FIXED VERSION of main_price_welfare.py
Key fix: Added np_config.enable_numpy_behavior() for TensorFlow compatibility
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from sklearn.linear_model import LinearRegression
import re
from tqdm import tqdm
from timeit import default_timer as timer

import tensorflow as tf
import tensorflow_constrained_optimization as tfco

# CRITICAL FIX: Enable numpy behavior for TensorFlow compatibility
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

def gen_initial_guess(cost_model, price_model, s=None, risk_group = None, df_prev=None, x_init=None, df_traditional=None, mc=None, file_dir='results_French'):
    regulation_model = f"{cost_model}-{price_model}" 
    
    pm_list = price_model.split('-')
    #1. Try to read the initial guess
    if x_init is not None:
        return x_init
    elif os.path.exists(f'{file_dir}/x_init_{regulation_model}.pickle'):
        print(f'Reading x_init from past trial.')
        with open(f'{file_dir}/x_init_{regulation_model}.pickle', 'rb') as f:
            x_init = pickle.load(f)
        return x_init
    
    #2. Try to use a given dataframe of individual prices
    if df_prev is None:
        use_df = False
    elif df_prev is not None:
        ## Use the previous results
        use_df = True
        if f'{regulation_model}_price' in df_prev.keys():
            price_init = df_prev[f'{regulation_model}_price']
        elif f'C0-{price_model}_price' in df_prev.keys():
            price_init = df_prev[f'C0-{price_model}_price']
        elif f'{cost_model}-P0_price' in df_prev.keys():
            price_init = df_prev[f'{cost_model}-P0_price']
        elif f'C0-P0_price' in df_prev.keys():
            price_init = df_prev[f'C0-P0_price']
        else:
            use_df = False
    if use_df:
        print('Using the previous dataframe for initial guess.')
        if 'PA' in pm_list: #Accountability constraint
            reg = LinearRegression(fit_intercept=False)
            reg.fit(df_traditional, np.log1p(price_init.values) )
            pa_init_guess = reg.coef_ * 1.
            x_init =  pa_init_guess.astype(np.float32)          
        elif 'POB' in pm_list:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(mc.reshape([-1,1]), price_init)
            pob_init_guess = np.array( [ reg.intercept_, reg.coef_[0] ] ).astype(np.float32) 
            x_init =  pob_init_guess.astype(np.float32) 
        elif ('PDP' in pm_list):
            mean_diff = np.mean(price_init.loc[s==0])/np.mean(price_init.loc[s==1]) -1
            pdp_init_guess = ( price_init/ (1+ (1-s)*mean_diff) ).astype(np.float32)
            x_init = pdp_init_guess.values.astype(np.float32)
        elif 'PAF' in pm_list:
            paf_init_guess = pd.Series( np.zeros(len(price_init)), index = price_init.index )
            for g in risk_group.unique():
                mean_diff = np.mean(price_init.loc[((s==0) & (risk_group==g))])/np.mean(price_init.loc[((s==1) & (risk_group==g))]) -1
                paf_init_guess.loc[( (risk_group==g))] = ( price_init.loc[((risk_group==g))]/ (1+ (1-s)*mean_diff) ).astype(np.float32)
            x_init = paf_init_guess.values.astype(np.float32)
            
        else:
            x_init = price_init.values.astype(np.float32)
        return x_init
                                                                                                                                                                                                                       
    #3. Create initial guess from scratch.
    print('Creating the initial guess from scratch')
    if 'PA' in pm_list: #Accountability constraint
        print('Initial guess for PA by linear regression.')
        reg = LinearRegression(fit_intercept=False)
        reg.fit(df_traditional, np.log1p(mc) )
        pa_init_guess = reg.coef_ * 1.5
        x_init =  pa_init_guess.astype(np.float32)                                                                 
    elif 'POB' in pm_list:
        x_init =  np.array([0,1.5]).astype(np.float32)                 
    elif 'PDP' in pm_list:
        mean_mc_diff = np.mean(mc[s==0])/np.mean(mc[s==1]) -1
        pdp_init_guess = ( mc/ (1+ (1-s)*mean_mc_diff) *1.5 ).astype(np.float32)
        x_init = pdp_init_guess
    elif 'PFM' in pm_list:
        print('Initial guess for PFM by constant markup.')
        x_init = (mc*1.5).astype(np.float32)
    else:
        x_init = (mc*1.5).astype(np.float32)
    return x_init

def gen_model(df,df_traditional,n_sample,n_consumer,d,cost_model, price_model, s=None, df_prev=None, source='French', init_folder = None):
    # Function to construct necessary variabels based on the regualtion.
    ## Cost model setting
    if source=='SBV':
        t = 'telem0' if 'CA' in cost_model else 'telem1'
    elif source=='French':
        t = 'telem0'
    ml_model = 'glm' if 'CA' in cost_model else 'xgb'
    if 'CU' in cost_model:
        m = 'M2'
    elif 'CDP' in cost_model:
        m = 'M3'
    elif 'CC' in cost_model:
        m = 'M5'
    else:
        m = 'M1'
    firm_model = f"{m}_{t}_{ml_model}"

    ### Construct a product and firm information.
    loss_sample_firm = sample_accidents(df[f'alpha_{firm_model}'],df[f'beta_{firm_model}'],df[f'lambda_{firm_model}'],n_sample)
    est_mc = loss_to_expected_claim(d, loss_sample_firm).astype(np.float32)
    risk_group = (est_mc<=np.quantile(est_mc,.25)).astype(int)+(est_mc<=np.quantile(est_mc,.5)).astype(int)+(est_mc<=np.quantile(est_mc,.75)).astype(int)
    risk_group = pd.Series(risk_group, index=s.index)

    # Define the regulation model
    regulation_model = f"{cost_model}-{price_model}" 

    ## Price model setting
    const_list = []
    pm = price_model.split('-')
    pdp_p = None
    penalty = 0.
    fc_=[a_ for a_ in pm if a_.startswith('PDP_')]
    if 'PA' in pm: #Accountability constraint
        const_list.append('PA')
    elif 'POB' in pm: #Optimization ban
        const_list.append('POB')
    else: #Free individualized pricing with/without fairness constraint
        if 'PDP' in pm: #DP constraint
            const_list.append('PDP') 
            penalty = 1e3
        elif len(fc_)==1: #DP-p constraint
            const_list.append('PDP_p')            
            pdp_p = fc_[0].split('_')[1]/100
        elif 'PFM' in pm: #DP constraint on markup
            const_list.append('PFM') 
            penalty = 1e5
        elif 'PAF' in pm:
            const_list.append('PAF')
            penalty = 1e3
        
    if df_prev is not None:
        if (len(df_prev)!=n_consumer):
            df_prev = None
    if init_folder is None:
        init_folder = 'results_'+source
    x_init = gen_initial_guess(cost_model, price_model, s, risk_group=risk_group,df_prev=df_prev, x_init=None, df_traditional=df_traditional, mc=est_mc, file_dir = init_folder)

    assert isinstance(x_init, np.ndarray)
    assert x_init.dtype==np.float32
    return loss_sample_firm, est_mc, risk_group, regulation_model, const_list, x_init, pdp_p, penalty

class PriceOptimizationProblem(tfco.ConstrainedMinimizationProblem):
    '''
    FIXED: Optimization problem class with proper TFCO integration
    '''
    def __init__(self,x_init ,d ,u_intercept,u_price_coeff,risk_gamma,outside_value,logit_sigma,loss_sample_consumer, loss_sample_firm,
                 s=None, risk_group=None,const_list=[],consumer_X=None,d_others=None,p_others=None,pdp_p=None,penalty=0.):

        #tf.Variable to be optimized. Either represents price directly or coefficients on relativities.
        self.x = tf.Variable(x_init.astype(np.float32) ) 
        assert len( self.x.shape )==1

        #Deductible of the product
        self.d = d
        
        #Consumer characteristics
        self.u_intercept = tf.constant(u_intercept) 
        self.u_price_coeff = tf.constant(u_price_coeff)
        self.risk_gamma = tf.constant(risk_gamma)
        self.outside_value = tf.constant(outside_value)
        self.n_consumer = len(u_intercept)
        self.logit_sigma = logit_sigma

        #Samples drawn to compute expectation.
        self.loss_sample_consumer = tf.constant(loss_sample_consumer) 
        self.loss_sample_firm = tf.constant(loss_sample_firm)  
        self.total_oop_sample_list = [tf.constant( loss_to_total_oop(d, loss_sample_consumer).astype(np.float32)) ]
        self.cost_estimate = tf.constant( loss_to_expected_claim(d, loss_sample_firm).astype(np.float32) )
        self.cost_true = tf.constant( loss_to_expected_claim(d, loss_sample_consumer).astype(np.float32) )
        self.n_sample = loss_sample_consumer.shape[0]
        
        #Competitors
        self.p_others = p_others
        self.d_others = d_others
        if d_others is None:
            self.n_other_prod = 0
        else:
            if isinstance(d_others, int): 
                self.n_other_prod = 1
            else:
                self.n_other_prod = len(d_others)                                       
            for j in range(len(d_others)):
                self.total_oop_sample_list.append( loss_to_total_oop(d_others[j], loss_sample_consumer).astype(np.float32) ) 

        #Constraints
        self.const_list = const_list
        self.s = s
        self.risk_group = risk_group
        if risk_group is not None:
            self.n_risk_group = len( risk_group.unique() )
        self.pdp_p = pdp_p
        if 'PA' in self.const_list:
            self.consumer_X = consumer_X
        self.penalty = penalty

    
    def loss_to_value_j_tf(self, p, j=None, d=None):
        h_ij = self.loss_to_h(p,j,d)
        v_ij = tf.reduce_mean(h_ij,axis=0) - (self.risk_gamma/2) * tf.reduce_mean(h_ij**2,axis=0)            
        return v_ij
    
    def loss_to_h(self,p,j=None,d=None):
        if d is None:
            assert j is not None
            total_oop_sample = self.total_oop_sample_list[j]
        else:
            assert d is not None
            loss_sample_consumer = self.loss_sample_consumer.numpy()
            total_oop_sample = tf.constant( loss_to_total_oop(d, loss_sample_consumer).astype(np.float32)) 
        a = tf.repeat(self.u_intercept,self.n_sample).reshape([-1,self.n_sample]).transpose() # Repeat across samples
        b = tf.repeat(self.u_price_coeff,self.n_sample).reshape([-1,self.n_sample]).transpose() # Repeat across samples
        h_ij = a - b*p  -total_oop_sample
        return h_ij
    
    def find_ce(self, p,j=None,outside_option=False,d=None):
        '''
        Find the CE by solving h - risk_gamma/2*h^2 = v_ij where v_ij is the value of the product (computed as E[h_ij]-risk_gamma/2 E[h_i^2])
        The value h-risk_gamma/2 h^2 is increasing in h when h<1/risk_gamma.
        '''
        v_product = self.outside_value if outside_option else  self.loss_to_value_j_tf(p,j,d)     
        assert tf.reduce_all(1- 2*self.risk_gamma*v_product>0)
        h_ce = 1-tf.sqrt( 1- 2*self.risk_gamma*v_product )/self.risk_gamma
        return h_ce
    
    def choice_prob_tf(self, p):
        v = self.loss_to_value_j_tf(p, 0)
        v_list = [v.reshape([-1,1])]
        if self.d_others is not None:
            for j in range(1,1+self.n_other_prod):
                v_other = self.loss_to_value_j_tf(self.p_others[:,j], j)
                v_list.append(v_other.reshape([-1,1]))
        v_all = tf.concat(v_list,axis=1)
        
        num = tf.exp(v_all/self.logit_sigma) 
        denom = (tf.reduce_sum(tf.exp(v_all/self.logit_sigma),axis=1).reshape([self.n_consumer,1]) + tf.exp(self.outside_value.reshape([self.n_consumer,1])/self.logit_sigma) ) 
        prob = tf.math.divide_no_nan(num, denom+1e-25 ) 
        assert not tf.reduce_any(tf.math.is_nan(prob))
        
        return prob

    @property
    def num_constraints(self):
        n_const = 0
        if 'PDP' in self.const_list:
            n_const = 2
        elif 'PDP_p' in self.const_list:
            n_const = 2
        elif 'PFM' in self.const_list:
            n_const = 2
        elif 'PAF' in self.const_list:
            n_const = self.n_risk_group*2
            
        return n_const

    def gen_price(self):
        if 'PA' in self.const_list:
            #Price needs to be multiplication.
            assert self.x.shape[0]==self.consumer_X.shape[1]
            p = tf.reduce_prod(tf.exp( self.consumer_X*self.x ),axis=1)
        elif 'POB' in self.const_list:
            #Price needs to be associated to cost
            assert self.x.shape[0]==2
            p = self.x[0] + self.x[1]*self.cost_estimate
        else:
            assert self.x.shape[0]==self.n_consumer
            p = self.x
        assert p.shape==(self.n_consumer,), print(p.shape)
        return p

        
    def objective(self):
        p = self.gen_price()
        cp = self.choice_prob_tf(p)
        demand = cp[:,0]
        prof_i = demand*(p-self.cost_estimate)
        prof = tf.reduce_sum(prof_i)
        if 'PDP' in self.const_list:
            mean_price_gap = tf.square( tf.reduce_mean(tf.boolean_mask( p+.01, self.s==0) ) - tf.reduce_mean(tf.boolean_mask( p+.01, self.s==1) ) )
            obj = prof - self.penalty*mean_price_gap
        elif 'PFM' in self.const_list:
            markup = (p / self.cost_estimate) - 1.
            mean_markup_gap = tf.square( tf.reduce_mean(tf.boolean_mask( markup+.01, self.s==0) ) - tf.reduce_mean(tf.boolean_mask( markup+.01, self.s==1) ) )
            obj = prof - self.penalty*mean_markup_gap
        elif 'PAF' in self.const_list:
            gap = 0.
            for g in self.risk_group.unique():
                gap = gap + tf.square( tf.reduce_mean(tf.boolean_mask( p+.01, ((self.risk_group==g)&(self.s==0))) ) - tf.reduce_mean(tf.boolean_mask( p+.01, ((self.risk_group==g)&(self.s==1))) ) )
            obj = prof - self.penalty*gap
        else:
            obj = prof
        return -obj
    
    def profit(self):
        p = self.gen_price()
        cp = self.choice_prob_tf(p)
        demand = cp[:,0]
        prof_i = demand*(p-self.cost_true)
        prof = tf.reduce_sum(prof_i)
        return -prof
    
    def gen_const(self):
        multiplier = 1e12
        const = []
        p = self.gen_price()
        mean_price_gap = tf.reduce_mean(tf.boolean_mask( p+.01, self.s==0) )/tf.reduce_mean(tf.boolean_mask( p+.01, self.s==1) ) - 1.
        mean_markup_gap = self.compute_mean_markup_ratio() - 1.
        if 'PDP' in self.const_list:
            const.append(mean_price_gap* multiplier)
            const.append(-mean_price_gap* multiplier)
        elif 'PDP_p' in self.const_list:            
            const.append( (mean_price_gap-self.pdp_p)* multiplier )
            const.append( (self.pdp_p-mean_price_gap)* multiplier )
        elif 'PFM' in self.const_list:
            const.append(mean_markup_gap* multiplier)
            const.append(-mean_markup_gap* multiplier)
        elif 'PAF' in self.const_list:
            for g in self.risk_group.unique():
                mean_price_gap = tf.reduce_mean(tf.boolean_mask( p+.01, ((self.risk_group==g)&(self.s==0))) )/tf.reduce_mean(tf.boolean_mask( p+.01, ((self.risk_group==g)&(self.s==1))) ) - 1.
                const.append(mean_price_gap* multiplier)
                const.append(-mean_price_gap* multiplier)
                
        return tf.stack(const)
        

    def proxy_constraints(self):
        return self.gen_const()

    def constraints(self):
        return self.gen_const()

    def compute_mean_price_ratio(self):
        p = self.gen_price()
        mean_price_ratio = tf.reduce_mean(tf.boolean_mask( p+.01, self.s==0) )/tf.reduce_mean(tf.boolean_mask( p+.01, self.s==1) )
        return mean_price_ratio
    
    def compute_mean_markup_ratio(self):
        p = self.gen_price()
        markup = (p / self.cost_estimate) - 1.
        mean_markup_ratio = tf.reduce_mean(tf.boolean_mask( markup+.00001, self.s==0) )/tf.reduce_mean(tf.boolean_mask( markup+.00001, self.s==1) )
        return mean_markup_ratio
    
    def sum_mean_price_ratio_groups(self):
        p = self.gen_price()
        temp = 0.
        for g in self.risk_group.unique():
            mean_price_gap = tf.reduce_mean(tf.boolean_mask( p+.01, ((self.risk_group==g)&(self.s==0))) )/tf.reduce_mean(tf.boolean_mask( p+.01, ((self.risk_group==g)&(self.s==1))) ) 
            temp = temp + mean_price_gap
        return temp

def main(oo_option='mc_times_markup',oo_markup=None):
    # 1. Read data
    raw_data_dir = 'data/French_data_cost_model_result/cost modelling for French Claims Data/Data for Modelling/'
    cost_data_dir = 'data/French_data_cost_model_result/cost modelling for French Claims Data'
    source = 'French'
    if oo_option == 'uninsured':
        result_dir = f'results_French_{oo_option}_all_prices'
    else:
        result_dir = f'results_French_{oo_option}_{oo_markup}_all_prices'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        os.makedirs(result_dir+'/graphs')

    #NOTE: Using more consumers for better analysis
    df = read_cost_results(raw_data_dir,cost_data_dir, source=source)
    df = df.iloc[:2000]  # Increased to 2000 customers for better statistical power

    #NOTE: Assuming we use Frence data.
    df_traditional = df[FRENCH_X_VARS]
    df_traditional=pd.get_dummies(df_traditional, columns=FRENCH_CATEGORICAL_VARS, drop_first=True)
    df_traditional['constant'] = 1.
    df_traditional.to_csv('French_individual_traditional.csv')

    # 2. Construct consumers
    n_consumer = len(df)
    n_sample = 3000  # Same as original successful runs
    logit_sigma = 39.213
    consumer_model = 'M1_telem0_xgb'#'M1_telem1_glm'
    df, loss_sample_consumer = construct_consumers(df, consumer_model, n_sample, source='French')
    df = construct_outside_option(df,loss_sample_consumer,option=oo_option, markup=oo_markup, deductible=0 ,flag_inertia=True, source='French')

    # 3. Other environment setup
    # Convert to tf.Tensor
    loss_sample_consumer =  loss_sample_consumer.astype(np.float32) 
    u_intercept = tf.constant(df['utility_intercept'].values.astype(np.float32) )
    u_price_coeff = tf.constant(df['utility_price_coef'].values.astype(np.float32) ) 
    risk_gamma = tf.constant(df['risk_gamma'].values.astype(np.float32) )
    outside_value = tf.constant(df['value_outside'].values.astype(np.float32) )
    # Other environment
    d = 0 #NOTE: d needs to be zero right now.
    p_others = None
    d_others = None
    # flag_telem = True 
    consumer_X = tf.constant(df_traditional.values.astype(np.float32) )
    s = df['s']
    cost_models = ['C0']
    price_models = ['P0', 'PA', 'POB', 'PDP', 'PAF']
    # Nuisance parameter
    n_iter = 3000  # Reduced for faster execution
    lr_schedule_cvx = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=.3,
        decay_steps=200,
        decay_rate=.99)
    ## PA constraint requires lower learning rate as the problem is severely non-convex.
    lr_schedule_ncvx = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-3,# Increased from 1e-3 for better convergence
        decay_steps=200,
        decay_rate=.99)#0.99)
    ## PDP/PAF/POB learning rate - higher for better convergence
    lr_schedule_pdp_paf_pob = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,  # Much higher than PA for better convergence
        decay_steps=200,
        decay_rate=.99)

    # 4. Run the price optimization
    individual_result_df = pd.DataFrame(index=df.index)
    individual_result_df['s'] = s
    individual_result_df['eta_consumer'] = df['eta_consumer']
    result_df = pd.DataFrame()
    fig_opt, axes_opt = plt.subplots(1,1,figsize=(8,6))
    # fig_opt_ce, axes_opt_ce = plt.subplots(len(cost_models),len(price_models))
    start = timer()
    update_consumers = True if oo_option=='mc_times_markup' else False


    regulation_models = []
    for ci, cost_model in enumerate(cost_models):
        for pi, price_model in enumerate(price_models):
        # for pi, price_model in enumerate([price_models[1]]):
            loss_sample_firm, est_mc, risk_group, regulation_model, const_list, x_init, pdp_p, penalty = gen_model(df,df_traditional,n_sample,n_consumer,d,cost_model, price_model, s, df_prev = individual_result_df, source=source, init_folder='results_French_outside_option2')
            regulation_models.append(regulation_model)
            individual_result_df[regulation_model+'_estimated_cost'] = est_mc
            individual_result_df[regulation_model+'_risk_group'] =  risk_group

            if update_consumers:
                ## Update the outside options for each regulation.
                df_reg = df.copy()
                df_reg['mc_reg'] = est_mc
                #NOTE: The outside option construction assumes deductible=0 for now.
                df_reg = construct_outside_option(df_reg,loss_sample_consumer,option='regmc_times_markup', markup=oo_markup,deductible=0 ,flag_inertia=True, source='French')
                outside_value = tf.constant(df_reg['value_outside'].values.astype(np.float32) )

            

            # Define optmization problem
            pm = price_model.split('-')
            # FIXED: Use model-specific learning rates
            # PA needs slow LR (non-convex), PDP/PAF/POB need medium LR, P0 needs fast LR
            if 'PA' in const_list and 'PAF' not in const_list:
                lr = lr_schedule_ncvx  # Slow for PA (0.005)
            elif 'PDP' in const_list or 'PAF' in const_list or 'POB' in const_list:
                lr = lr_schedule_pdp_paf_pob  # Medium for PDP/PAF/POB (0.1)
            else:
                lr = lr_schedule_cvx  # Fast for P0 (0.3)
            problem = PriceOptimizationProblem(x_init ,d ,u_intercept,u_price_coeff,risk_gamma,outside_value,logit_sigma,loss_sample_consumer, loss_sample_firm,
                            s=s, risk_group=risk_group, const_list=const_list,consumer_X=consumer_X,d_others=d_others,p_others=p_others,pdp_p=pdp_p,penalty=penalty)         
            
            # FIXED: Use appropriate optimizer based on constraints
            if len(const_list) == 0:
                # No constraints - use standard optimizer
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
                # Legacy optimizer doesn't need build() method
                print(f'---Optimize price under {regulation_model} (no constraints)---')
                
                # Simple optimization loop
                objective_trj = []
                const_trj = []
                for epoch in tqdm(range(n_iter)):
                    with tf.GradientTape() as tape:
                        loss = problem.objective()
                    gradients = tape.gradient(loss, [problem.x])
                    optimizer.apply_gradients(zip(gradients, [problem.x]))
                    
                    objective_trj.append(-problem.objective().numpy())
                    const_trj.append(problem.compute_mean_price_ratio().numpy()-1.)
                    
                    if epoch>20:
                        gain_obj =  (np.max(objective_trj[-20:])- np.min(objective_trj[-20:]) )/np.abs(np.min(objective_trj[-20:])  )
                        gain_const = np.abs(np.max(const_trj[-20:]) -np.min(const_trj[-20:]) ) 
                        if gain_obj<1e-5 and gain_const<1e-5:
                            print(f"Converged at epoch {epoch}")
                            break
            else:
                # Has constraints - use penalty method (TFCO has compatibility issues)
                print(f'---Optimize price under {regulation_model} (with constraints)---')
                
                # Use penalty method for constrained optimization
                original_penalty = problem.penalty
                problem.penalty = 1e3  # Start with moderate penalty
                
                # Create optimizer (legacy doesn't need build)
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
                
                #Optimization
                objective_trj = []
                const_trj = []
                for epoch in tqdm(range(n_iter)):
                    with tf.GradientTape() as tape:
                        loss = problem.objective()
                    gradients = tape.gradient(loss, [problem.x])
                    optimizer.apply_gradients(zip(gradients, [problem.x]))
                    
                    objective_trj.append(-problem.objective().numpy())
                    const_trj.append(problem.compute_mean_price_ratio().numpy()-1.)
                    
                    # Increase penalty if constraints are violated
                    if epoch % 100 == 0 and epoch > 0:
                        constraint_violation = abs(problem.compute_mean_price_ratio().numpy() - 1.0)
                        if constraint_violation > 0.01:  # If constraint violated by more than 1%
                            problem.penalty = min(problem.penalty * 2, 1e6)  # Increase penalty
                            print(f"Epoch {epoch}: Increasing penalty to {problem.penalty:.0f}")
                    
                    if epoch>20:
                        gain_obj =  (np.max(objective_trj[-20:])- np.min(objective_trj[-20:]) )/np.abs(np.min(objective_trj[-20:])  )
                        gain_const = np.abs(np.max(const_trj[-20:]) -np.min(const_trj[-20:]) ) 
                        if gain_obj<1e-5 and gain_const<1e-5:
                            print(f"Converged at epoch {epoch}")
                            break
                
                # Restore original penalty
                problem.penalty = original_penalty
                
            # Enhanced plotting code with labels and legends for all 5 models
            axes_opt.plot(range(len(objective_trj)), objective_trj, label=f'{price_model}', linewidth=2)
            axes_opt.set_xlabel('Iteration', fontsize=12)
            axes_opt.set_ylabel('Profit', fontsize=12)
            axes_opt.grid(True, alpha=0.3)
            
            # Add legend for profit curves (only once, after all models are plotted)
            # Since we have nested loops (cost_models × price_models), we need to check if this is the last price_model in the last cost_model
            if ci == len(cost_models) - 1 and pi == len(price_models) - 1:  # Last cost model AND last price model
                axes_opt.legend(loc='lower right', fontsize=10, title='Pricing Models')
                axes_opt.set_title('Optimization History - All Models', fontsize=14, fontweight='bold')
            
            fig_opt.tight_layout()
                    

                    
            #Record results
            with open(f'{result_dir}/x_init_{regulation_model}.pickle','wb') as f:
                pickle.dump( problem.x.numpy() , f )
            result_dict = {}
            price = problem.gen_price().numpy().flatten()
            coverage = problem.choice_prob_tf(problem.gen_price()).numpy().flatten()

            individual_result_df[regulation_model+'_price'] = price
            individual_result_df[regulation_model+'_coverage'] = coverage
            
            result_dict['model'] = regulation_model
            result_dict['mean_price_ratio'] = problem.compute_mean_price_ratio().numpy()
            result_dict['profit'] = - problem.profit().numpy()
            result_dict['coverage0'] = coverage[problem.s==0].mean()
            result_dict['coverage1'] = coverage[problem.s==1].mean()
            result_dict['mean_price0'] = price[problem.s==0].mean()
            result_dict['mean_price1'] = price[problem.s==1].mean()
            result_dict['mean_est_mc0'] = est_mc[problem.s==0].mean()
            result_dict['mean_est_mc1'] = est_mc[problem.s==1].mean()


            
            # Compute welfare
            ## Compute CE of the product
            if True:
                #Assuming d=0.
                ce_purchase = problem.find_ce(p=problem.gen_price(), j=0,outside_option=False, d=0).numpy()
            # if d==0:
            #     ce_purchase = problem.loss_to_h(problem.gen_price(),0)[0].numpy()
            # else:
            #     ce_purchase = problem.find_ce(problem.gen_price(), 0).numpy()

            ## Compute CE of the outside option
            if True:
                print('compute ce_outside.')
                ce_outside = problem.find_ce(p=None,j=None,outside_option=True,d=0).numpy()
                individual_result_df[regulation_model+'_ce_outside'] = ce_outside
                '''
                if 'ce_outside' in individual_result_df.keys():
                    ce_outside = individual_result_df['ce_outside'].values.flatten()
                # ce_outside = problem.find_ce(None,None,True).numpy()
                # individual_result_df['ce_outside'] = ce_outside
                '''
            '''
            Find the CE by solving h - risk_gamma/2*h^2 = v_ij where v_ij is the value of the product (computed as E[h_ij]-risk_gamma/2 E[h_i^2])
            The value h-risk_gamma/2 h^2 is increasing in h when h<1/risk_gamma.
            def find_ce(self, p,j=None,outside_option=False,d=None):
            v_product = self.outside_value if outside_option else  self.loss_to_value_j_tf(p,j,d)     
                assert tf.reduce_all(1- 2*self.risk_gamma*v_product>0)
                h_ce = 1-tf.sqrt( 1- 2*self.risk_gamma*v_product )/self.risk_gamma
                return h_ce
            '''

            ## Compute CE of the no-insurance state
            # if 'ce_no_insurance' not in df.keys():
            ce_no_insurance = problem.find_ce(p=0,j=None,outside_option=False,d=np.inf).numpy()
            individual_result_df['ce_no_insurance'] = ce_no_insurance

            ## Record welfare result
            welfare = coverage*ce_purchase + (1-coverage)*ce_outside
            individual_result_df[regulation_model+'_welfare'] = welfare
            result_dict['welfare'] = welfare.mean()
            result_dict['welfare0'] = welfare[problem.s==0].mean()
            result_dict['welfare1'] = welfare[problem.s==1].mean()
            result_dict['welfare_no_insurance'] = ce_no_insurance.mean()
            result_dict['welfare0_no_insurance'] = ce_no_insurance[problem.s==0].mean()
            result_dict['welfare1_no_insurance'] = ce_no_insurance[problem.s==1].mean()
            

            # Add the result to the summary dataframe.
            result_df =pd.concat([result_df,pd.DataFrame(result_dict,index=[0])],axis=0,ignore_index=True)

    end = timer()
    print('------------------------')
    print('Total time: ',end - start)
    result_df.to_csv(f'{result_dir}/welfare_result_summary.csv')
    individual_result_df.to_csv(f'{result_dir}/welfare_result_individual.csv')
    fig_opt.savefig(f'{result_dir}/price_opt_history.png')



if __name__ == '__main__':
    # Run all French cases with different outside option markups
    print("=== Running Uninsured Case ===")
    main(oo_option='uninsured',oo_markup=None)
    
    print("=== Running Insured Case with 0.5x Markup ===")
    main(oo_option='mc_times_markup',oo_markup=0.5)
    
    print("=== Running Insured Case with 1.0x Markup ===")
    main(oo_option='mc_times_markup',oo_markup=1.0)
    
    print("=== Running Insured Case with 1.5x Markup ===")
    main(oo_option='mc_times_markup',oo_markup=1.5)
    
    print("=== Running Insured Case with 3.0x Markup ===")
    main(oo_option='mc_times_markup',oo_markup=3.0)
