'''
Testing some basic functions for the demand/pricing model.
'''
import pandas as pd
import numpy as np
from numpy.random import gamma, poisson
# from scipy.optimize import minimize
# import cvxpy as cp

# import jax.numpy as jnp
# from jax import grad, jit, vmap
# from jax import random
# from jax import jacfwd, jacrev
import tensorflow as tf
import tensorflow_constrained_optimization as tfco

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


SBV_TRADITIONAL_VARS = ['Duration', 'Insured.age', 'Insured.sex', 'Car.age', 'Marital',
       'Car.use', 'Credit.score', 'Region', 'Territory',
       'Years.noclaims','Annual.miles.drive']
SBV_CATEGORICAL_VARS = ['Insured.sex','Marital','Car.use','Region', 'Territory']

FRENCH_X_VARS = ['Age.ct', 'Bonus', 'Density', 'GroupOne', 'Insurancescore', 'Gender']
FRENCH_CATEGORICAL_VARS = ['Gender']

ADDITIONAL_DATA_X_VARS = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'vh_speed', 'vh_value',
       'pol_coverage', 'drv_sex1', 'vh_fuel', 'vh_makenew', 'pol_bonus',
       'drv_age_lic1', 'pol_duration', 'pol_sit_duration']
ADDITIONAL_DATA_CATEGORICAL_VARS = ['drv_sex1', 'vh_fuel', 'vh_makenew', 'pol_coverage']

def sample_accidents(a,b,l,n_sample):
    '''
    Sample sequences of accidents from compound Poisson-Gamma distribution.
    parameters:
      a,b: parameters of gamma(a,1/b). Note that b is assumed to be the rate while np.random.gamma takes the scale. array of the shape (n_consumer).
      l: parameter poisson(l). array of the shape (n_consumer).
      n_sample: number of samples to draw
    return: Sequence of accident sizes of the shape (n_sample, max(n_accidents), n_consumer). Size is set to be zero if no further accidents.
    '''
    n_consumer = len(a)
    n_accident = poisson(l,size=(n_sample,n_consumer))
    max_n_accident = np.max(n_accident)
    size = gamma(a,1/b,size=(n_sample,max_n_accident,n_consumer) )

    mask = np.zeros_like(size)
    for i in range(n_consumer):
        n_accident_i = n_accident[:,i]
        for j in range(n_sample):
            n = n_accident_i[j]
            mask[j,:n,i] = 1.
    loss_sample = size * mask

    # loss_sample = np.array([np.lib.pad(size[i,:n_accidents[i]], (0, max_n_accidents - n_accidents[i]), 'constant', constant_values=0) for i in range(n_sample)])
    return loss_sample


def loss_to_total_claim(d, loss_sample):
    '''
    Return total claim amount from the samples of losses.
    parameters:
      d: deductible. The claim amount is max(z-d,0), where z is the loss size.
      loss_sample: array of the shape (#sample,max #accidents, n_consumer).
    return: sample of total claim amount of shape (n_sample, n_consumer)
    '''
    claim_sample = np.maximum(loss_sample-d,0)
    total_claim_sample = np.sum(claim_sample,axis=1)
    return total_claim_sample

def loss_to_expected_claim(d, loss_sample):
    '''
    Compute expected claim amount from the samples of losses.
    This serves as the marginal cost of products-consumers for the firm.
    parameters:
      d: deductible. The claim amount is max(z-d,0), where z is the loss size.
      loss_sample: Samples of loss size sequence of accidents. array of the shape (#sample,max #accidents,n_consumer).
    return: sample of total claim amount of shape (n_consumer)
    '''
    claim_sample = np.maximum(loss_sample-d,0)
    total_claim_sample = np.sum(claim_sample,axis=1)
    expected_total_claim = np.mean(total_claim_sample, axis=0)
    return expected_total_claim


def loss_to_total_oop(d, loss_sample):
    '''
    Return total out-of-pocket expense from the samples of losses.
    parameters:
      d: deductible. The claim amount is max(z-d,0), where z is the loss size.
      loss_sample: Samples of loss size sequence of accidents. array of the shape (#sample,max #accidents,n_consumer).
    return: sample of oop expense of shape (n_sample,)
    '''
    is_accident = (loss_sample>0).astype(float)
    oop_sample = np.minimum(loss_sample,d) * is_accident
    total_oop_sample = np.sum(oop_sample,axis=1)
    return total_oop_sample

# def loss_to_value_j(p,d,eta,is_stay,risk_gamma,loss_sample):
def loss_to_value_j(p,d,u_intercept,u_price_coeff,risk_gamma,loss_sample):
    '''
    Expected value of a product j for a consumer. Following Cohen-Einav and Jin-Vasserman, the value is computed as
      v_ij=E[u_ij]=E[h_ij]-gamma/2E[h_ij^2]
      h_ij = -p_ij - out-of-pocket.
    Parameters:
      p: The price of the product j. Array of shape (n_consumer).
      d: Deductible amount of the product j. Scalar.
      u_intercept: base value of the product.
      u_price_coeff: price coefficient in value.
      risk_gamma: risk aversion coefficient. Scaler (as assumed in JV) or (n_consumer).
      loss_sample: Samples of loss size sequence of accidents. array of the shape (#sample,max #accidents,n_consumer).
    return:
      v_ij: The value of a product j for consumers. Array of shape (n_consumer).
    '''

    #Make sure everything is 1D array.
    if isinstance(p, pd.Series):
        p = p.values
    if isinstance(u_intercept, pd.Series):
        u_intercept = u_intercept.values
    if isinstance(u_price_coeff, pd.Series):
        u_price_coeff = u_price_coeff.values
    if isinstance(risk_gamma, pd.Series):
        risk_gamma = risk_gamma.values
    u_intercept = u_intercept.flatten()
    u_price_coeff = u_price_coeff.flatten()
    risk_gamma = risk_gamma.flatten()
    p = p.flatten()

    total_oop_sample = loss_to_total_oop(d, loss_sample)
    n_sample = loss_sample.shape[0]
    # h_ij = -p -is_stay*eta -total_oop_sample
    a = u_intercept.repeat(n_sample).reshape([-1,n_sample]).transpose() # Repeat across samples
    b = u_price_coeff.repeat(n_sample).reshape([-1,n_sample]).transpose() # Repeat across samples
    h_ij = a - b*p  -total_oop_sample
    v_ij = np.mean(h_ij,axis=0) - (risk_gamma/2) * np.mean(h_ij**2,axis=0)
    return v_ij

# def loss_to_value_all_prod(p_array, d_array, eta, is_stay,risk_gamma,loss_sample):
def loss_to_value_all_prod(p_array, d_array, u_intercept, u_price_coeff,risk_gamma,loss_sample):
    '''
    Expected value of products in the market.
    Parameters:
      p_array: price matrix of shape (n_consumer,n_product), or an array (n_consumer)
      d_array: deductible of products. array of shape (n_product), or a float.
    Return: (n_consumer, n_product) array of values.
    '''
    n_consumer = len(u_intercept)
    n_product = 1 if not isinstance(d_array, np.ndarray) else len(d_array)
    v = np.zeros([n_consumer,n_product])

    if n_product==1:
        p_array = p_array.reshape([-1,1])
    if not isinstance(d_array, np.ndarray):
        d_array = np.array([d_array])

    for j in range(n_product):
        p = p_array[:,j]
        d = d_array[j]
        #v[:,j] = loss_to_value_j(p,d,eta,is_stay,risk_gamma,loss_sample)
        v[:,j] = loss_to_value_j(p,d,u_intercept,u_price_coeff,risk_gamma,loss_sample)
    return v

def choice_prob(p_array, d_array, outside_value,u_intercept, u_price_coeff, logit_sigma,risk_gamma,loss_sample):
    '''
    Compute choice probability from the value of products.
      p_array: price of products individualized for each consumer. (n_consumer,n_product) or (n_consumer)
      d_array: deductible of products. (n_product) or scalar if only one product.
      outside_value: value of outside option. (n_consumer).
      u_intercept: intercept term of utility. (n_consumer).
      u_price_coeff: price coefficient in utility. (n_consumer).
      logit_sigma: sigma of type-I EV error. Scalar.
      risk_gamma: absolute risk aversion coefficient. (n_consumer) or scalar.
      loss_sample: Samples of loss size sequence of accidents. array of the shape (#sample,max #accidents,n_consumer).
    Return:
      prob: Choice probability of products. (n_consumer,n_product)
    '''

    n_consumer = len(u_intercept)
    n_product = 1 if isinstance(d_array, int) else len(d_array)
    if n_product==1:
        p_array = p_array.reshape([-1,1])
    if not isinstance(d_array, np.ndarray):
        d_array = np.array([d_array])
    if isinstance(outside_value, pd.Series):
        outside_value = outside_value.values

    v = loss_to_value_all_prod(p_array, d_array, u_intercept, u_price_coeff, risk_gamma, loss_sample)
    prob = np.exp(v/logit_sigma)/(np.sum(np.exp(v/logit_sigma),axis=1).reshape([n_consumer,1]) + np.exp(outside_value.reshape([n_consumer,1])/logit_sigma) )

    return prob

# def compute_ce(p, d, alpha, beta,risk_gamma,loss_sample):
#     v = loss_to_value_j(p,d,alpha,beta,risk_gamma,loss_sample)
#     def diff_value(x):
#         v_hypo = loss_to_value_j(x,0,alpha,beta,risk_gamma,loss_sample)
#         return np.sum( (v-v_hypo)**2 )
#     sol = minimize(diff_value,v)


#     pass



"""
def opt_price_scipy(df, deductible, loss_sample_firm, loss_sample_consumer,logit_sigma,const,Xcols=None, penalty_strength=None):
    '''
    Find the optimal price to maximize the profit given constraints.
    Note: Currently assuming a single product.
    Inputs:
      df: Main dataframe.
      deductible: the deductible amount of the product.
      loss_sample_firm: sampled accidents from the cost model by firm, which may be restricted by accountability/fairness.
      loss_sample_consumer: sampled accidents from the cost model by consumers.
      const: list of constraints.
        - 'PM' requires the price to be multiplicable of factors.
        - 'PAFE-continuous'imposes a penalty on the average price difference between groups. penalty_strength needs to be specified.
        - 'PAFE' requires the average price to be equal between groups.
    '''
    n_consumer = len(df)
    n_product = 1 if isinstance(deductible, int) else len(deductible)
    def neg_profit(x):
        '''
        Assumes single product. Negative profit to be an objective of minimization problem.
        '''
        if 'PM' not in const:
            price = x.reshape([-1,n_product])
        elif 'PM' in const:
            #Price needs to be multiplication.
            price = np.prod( df[Xcols] * x , axis=1).reshape([-1,n_product])


        #choice_prob(p_array, d_array, outside_value,u_intercept, u_price_coeff, logit_sigma,risk_gamma,loss_sample)
        demand = choice_prob(price,deductible, df['value_outside'].values,df['utility_intercept'].values, df['utility_price_coef'].values,logit_sigma,df['risk_gamma'].values,loss_sample_consumer)
        cost = loss_to_expected_claim(deductible, loss_sample_firm).reshape([-1,n_product])
        prof_i = demand*(price-cost)
        prof = np.sum(prof_i)
        price_gap = np.mean(price[df['s']==0]) - np.mean(price[df['s']==1])

        if 'PAFE-continuous' in const:
            prof = prof - penalty_strength*np.abs(np.mean(price[df['s']==0]) - np.mean(price[df['s']==1]))

        return -prof

    def actuarial_fairness_in_expectation(p):
        price = p.reshape([-1,1])
        price_gap = np.mean(price[df['s']==0]) - np.mean(price[df['s']==1])
        return price_gap

    if 'PM' in const:
        x_init = np.ones(len(X_col))
    else:
        x_init = df['mc_d0'].values*1.1


    if 'PAFE' in const:
        sol = minimize(neg_profit,x_init,constraints={'type':'eq', 'fun': actuarial_fairness_in_expectation})
    else:
        sol = minimize(neg_profit,x_init)
    return sol
"""




def compute_CE_risk_gamma(df, source='SBV'):
    '''
    Assign the gamma coefficient of absolute risk aversion to consumers based on Cohen-Einav estimates.
    First we generate the heterogeneity using the variables overlappning across the source data and Cohen-Einav.
    - For SBV synthetic data, we use the coefficients of age, age^2, marital, young-driver.
    - For French claims data, we use age, age^2, young-driver.
    Then compute the intercept by matching the median.

    '''
    CE_coeff = pd.Series({'age':-.0623,'age_sq':6.44e-4,'young':-.2499,'female':.2049,'married':.1927} ) #From Table 4
    CE_median = 2.6e-5 #From Table 5

    if source=='SBV':
        age = df['Insured.age']
        female = (df['Insured.sex']=='Female').astype(float)
        married = (df['Marital']=='Married').astype(float)
    elif source=='French':
        age = df['Age.ct']
        female = (df['Gender']=='Female').astype(float)
        married = 0.
    elif source=='additional_data':
        age = df['drv_age1']
        female = (df['drv_sex1']=='F').astype(float)
        married = 0.


    df_CE = pd.DataFrame(index=df.index)
    df_CE['age'] = age#df['Insured.age']
    df_CE['age_sq'] = age**2#df['Insured.age']**2
    df_CE['young'] = (age<25).astype(float)#(df['Insured.age']<25).astype(float)
    df_CE['female'] = female#(df['Insured.sex']=='Female').astype(float)
    df_CE['married'] = married#(df['Marital']=='Married').astype(float)

    df_CE = df_CE.reindex(sorted(df_CE.columns), axis=1)
    CE_coeff = CE_coeff.reindex((CE_coeff.keys()),axis=1)
    log_risk_gamma_hetero= df_CE.mul(CE_coeff).sum(axis=1)

    #Normalize the coefficient so that the median is 
    log_risk_gamma_intercept  = np.log(CE_median) - np.median(log_risk_gamma_hetero)
    log_risk_gamma = log_risk_gamma_intercept+log_risk_gamma_hetero
    risk_gamma = np.exp(log_risk_gamma)
    return risk_gamma

def convert_df_SBV_to_JV(df, source='SBV'):
    '''
    Convert the variables to Jin and Vasserman to compute the inertia heterogeneity.

    '''
    df_new = pd.DataFrame(index=df.index)
    df_new['constant'] = 1.
    if source=='SBV':
        age = df['Insured.age']
        df_new['female'] = (df['Insured.sex']=='Female').astype(float)
        df_new['model_year'] = 2014-df['Car.age']
        df_new['density'] = 0.
        df_new['clean_record'] = (df['Years.noclaims']>df['Car.age']).astype(float)
    elif source=='French':
        age = df['Age.ct']
        df_new['female'] = (df['Gender']=='Female').astype(float)
        df_new['model_year'] = 0.
        df_new['density'] = df['Density']
        df_new['clean_record'] = 0.
    elif source=='additional_data':
        age = df['drv_age1']
        df_new['female'] = (df['drv_sex1']=='F').astype(float)
        df_new['model_year'] = df['vh_age']
        df_new['density'] = 0.
        df_new['clean_record'] = 0.


    df_new['age'] = age
    df_new['age_sq'] = age**2
    df_new['age<25'] = (age<25).astype(float)
    df_new['age>21'] = (age>21).astype(float)
    df_new['age>60'] = (age>60).astype(float)

    return df_new

def compute_JV_eta(df_JV):
    JV_eta_coeff=pd.Series(data={'constant':0,'age':4.526,'age_sq':3.816,'age<25':-.5,'age>21':3.195,'age>60':-.275,'female':1.007,'model_year':3.211,'clean_record':-1.392,'density':-4.902})

    ##Make sure the columns are aligned.
    df_JV = df_JV.reindex(sorted(df_JV.columns), axis=1)
    JV_eta_coeff = JV_eta_coeff.reindex(sorted(JV_eta_coeff.keys()),axis=1)

    eta=df_JV.mul(JV_eta_coeff).sum(axis=1)

    #Match the range of eta to Table 5.
    eta_min = 158
    eta_max = 407
    eta = (eta-eta.min() )/(eta.max()-eta.min()) * (eta_max-eta_min) + eta_min
    return eta


def construct_consumers(df, consumer_model, n_sample, source='SBV'):
    '''
    Construct simulated consumers from the data.
    Specifically, compute i) subjective risk profile, ii) risk coefficient, iii) value of outside option, iv) samples of accidents
      df: The main DataFrame.
      consumer_model: A model that consumers use to estimate their (lambda,alpha,beta) of accidents.
      n_sample: Number of samples to draw for accident simulation
      outside_option: Outside option to consider. If 'uninsured', outside option is not having any insurance. Instead it can specify {price:[], deductible:} where price is a (n_consumer) array and deductible is a scalar.
      flag_inertia: If True, the value of outside option is deducted by the inertia term computed from JV paper.
    Return:
      df: New DataFrame that also contain consumers' model
      loss_sample_consumer: Simulated accident sizes. (n_sample, n_max_accidents, n_consumer).
    '''


    n_consumer = len(df)

    df['lambda_consumer'] = df[f'lambda_{consumer_model}']
    df['alpha_consumer'] = df[f'alpha_{consumer_model}']
    df['beta_consumer'] = df[f'beta_{consumer_model}']
    df['risk_gamma'] = compute_CE_risk_gamma(df, source=source)

    df['utility_intercept'] = 0.
    df['utility_price_coef'] = 1.

    df['s'] = 0
    if source=='SBV':
        df.loc[df['Insured.sex']=='Female','s']=1
    elif source=='French':
        df.loc[df['Gender']=='Female','s']=1
    elif source=='additional_data':
        df.loc[df['drv_sex1']=='F','s']=1


    loss_sample_consumer = sample_accidents(df['alpha_consumer'],df['beta_consumer'],df['lambda_consumer'],n_sample)

    df['mc_d0'] = loss_to_expected_claim(0, loss_sample_consumer)

    return df, loss_sample_consumer

def construct_outside_option(df,loss_sample_consumer,option='uninsured', markup=None, price=None, deductible=None ,flag_inertia=False, source='SBV'):
    n_consumer = len(df)

    if option=='uninsured':
        price = np.zeros(n_consumer)
        deductible = np.inf
    elif option=='mc_times_markup':
        price = (df['mc_d0'] * markup).values
        deductible = 0
    elif option=='regmc_times_markup':
        price = (df['mc_reg'] * markup).values
        deductible = 0

    total_oop_sample = loss_to_total_oop(deductible, loss_sample_consumer)

    n_sample = loss_sample_consumer.shape[0]
    a = df['utility_intercept'].values.repeat(n_sample).reshape([-1,n_sample]).transpose() # Repeat across samples
    b = df['utility_price_coef'].values.repeat(n_sample).reshape([-1,n_sample]).transpose() # Repeat across samples
    h_ij = a - b*price  -total_oop_sample
    if flag_inertia:
        # eta_const = 134.262+228.559
        # JV_eta_coeff=pd.Series(data={'constant':eta_const,'age':4.526,'age_sq':3.816,'age<25':-.5,'age>21':3.195,'age>60':-.275,'female':1.007,'model_year':3.211,'clean_record':-1.392})
        df_JV = convert_df_SBV_to_JV(df, source=source)
        df['eta_consumer'] = compute_JV_eta(df_JV)
        df['is_stay'] = 1
        h_ij = h_ij - (df['eta_consumer']*df['is_stay']).values.repeat(n_sample).reshape([-1,n_sample]).transpose()

    v_ij = np.mean(h_ij,axis=0) - (df['risk_gamma'].values/2) * np.mean(h_ij**2,axis=0)
    df['value_outside'] = v_ij
    if deductible==0:
        df['ce_outside'] = h_ij[0]# if deductible is zero, h_ij should be all the same across samples

    return df


def read_cost_results(raw_data_dir,cost_data_dir, source='SBV'):

    if source=='SBV':
        df = pd.read_csv(f'{raw_data_dir}/Synthetic Dataset of Driver Telematics/telematics_syn-032021.csv')
    elif source=='French':
        # df = pd.read_csv(f'{raw_data_dir}/French Claims Data/FrenchClaimsData.csv', index_col=0)
        df = pd.read_csv(f'{raw_data_dir}/Cost_Modeling_Severity_M1_M2_M5.csv', index_col=0) #This file includes insurance score.
    elif source=='additional_data':
        df = pd.read_csv(f'{raw_data_dir}/Code Scripts in R/Additional Data/pg17_XGB_YfM1_poisson.csv', index_col=0)


    #Merge all the cost estimates so far.
    if source=='SBV':
        for a in ['', 'XGB_']:
            for m in range(1,6):
                for telem in [0,1]:
                    if m==4 and telem==0: #Model4 requires telematics data as the "legit" features to bais.
                        pass
                    else:
                        s_telem = 'with_telematics' if telem==1 else 'without_telematics'
                        df_poisson = pd.read_csv(f'{cost_data_dir}/{a}YfM{m}_{s_telem}_poisson.csv',index_col=0)
                        df_poisson.index = df_poisson.index-1
                        df_gamma = pd.read_csv(f'{cost_data_dir}/{a}YsM{m}_{s_telem}_gamma.csv',index_col=0)
                        df_gamma.index = df_gamma.index-1
                        if a=='':
                            df[f'lambda_M{m}_telem{telem}_glm'] = df_poisson['lambda']/df['Duration'] * 365 #Adjust it to the annual level.
                            df[f'alpha_M{m}_telem{telem}_glm'] = df_gamma['alpha']
                            df[f'beta_M{m}_telem{telem}_glm'] = df_gamma['beta']
                        elif a=='XGB_':
                            df[f'lambda_M{m}_telem{telem}_xgb'] = df_poisson['lambda']/df['Duration'] * 365 #Adjust it to the annual level.
                            df[f'alpha_M{m}_telem{telem}_xgb'] = df_gamma['mean']
                            df[f'beta_M{m}_telem{telem}_xgb'] = 1. #for now.
    elif source=='French':
        telem = 0
        for a in ['GLM_', 'XGB_']:
            for m in range(1,6):
                df_poisson = pd.read_csv(f'{cost_data_dir}/{a}YfM{m}_poisson.csv',index_col=0)
                df_gamma = pd.read_csv(f'{cost_data_dir}/{a}YsM{m}_gamma.csv',index_col=0)

                if a=='GLM_':
                    df[f'lambda_M{m}_telem{telem}_glm'] = df_poisson['lambda (predicted/Exppdays*365)']#Duration is already adjusted to annual level
                    df[f'alpha_M{m}_telem{telem}_glm'] = df_gamma['alpha']
                    df[f'beta_M{m}_telem{telem}_glm'] = df_gamma['beta']
                elif a=='XGB_':
                    df[f'lambda_M{m}_telem{telem}_xgb'] = df_poisson['lambda (predicted/Exppdays*365)']#Duration is already adjusted to annual level
                    df[f'alpha_M{m}_telem{telem}_xgb'] = df_gamma['mean']
                    df[f'beta_M{m}_telem{telem}_xgb'] = 1. #for now.
        df = df.dropna()
    elif source=='additional_data':
        telem = 0
        df_poisson = pd.read_csv(f'{raw_data_dir}/Code Scripts in R/Additional Data/pg17_XGB_YfM1_poisson.csv',index_col=0)
        df_gamma = pd.read_csv(f'{raw_data_dir}/Code Scripts in R/Additional Data/pg17_XGB_YsM1_gamma.csv',index_col=0)

        df = df_poisson.copy()
        df[f'lambda_M{1}_telem{0}_xgb'] = df_poisson['lambda (predicted/annual_exposure)']  # Duration is already adjusted to annual level
        df[f'alpha_M{1}_telem{0}_xgb'] = df_gamma['mean']
        df[f'beta_M{1}_telem{0}_xgb'] = 1.  # for now.


    return df




if __name__ == '__main__':
    n_sample = 100
    risk_gamma = 1.
    eta_const = 0.
    JV_eta_coeff=pd.Series(data={'constant':eta_const,'age':4.526,'age_sq':3.816,'age<25':-.5,'age>21':3.195,'age>60':-.275,'female':1.007,'model_year':3.211,'clean_record':-1.392})
    df = pd.DataFrame({'s':[1,1,1,0,0],'a':[1,2,3,4,5],'b':[1,2,1,2,1],'l':[1,1,1,1,1],'eta':[1,2,3,2,1],'is_stay':[1,1,1,1,1]})
    df['alpha'] = - df['eta']*df['is_stay']
    df['beta'] = 1.
    loss_sample = sample_accidents(df['a'],df['b'],df['l'],n_sample)
    price= np.array([1,2,1,1,.5])
    deductible= np.array([.3])

    # tes=minimize(neg_profit,np.ones(5),constraints={'type':'eq', 'fun': actuarial_fairness_in_expectation})

    JV_eta_coeff=pd.Series(data={'constant':134.262,'age':4.526,'age_sq':3.816,'age<25':-.5,'age>21':3.195,'age>60':-.275,'female':1.007,'model_year':3.211,'clean_record':-1.392})
