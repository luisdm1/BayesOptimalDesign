import pandas as pd
import numpy as np
from numpy import random
from numpy import log
from numpy import exp
from numpy import sqrt
from scipy.integrate import odeint


class Generate_ode_lineage_data:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.days_data = data_dict["days_data"]
        self.days_data_idx = data_dict["days_data_ix"]
        self.N_time_points = len(self.days_data)
        self.log_data = data_dict["log_data"]

    def toy_lineage_ode(self, y, t, theta):
        #parameters
        p0max, eta1, eta2maxbar, gam1, gam5 = theta

        dydt = np.zeros(2)
        # Rescale gamma parameters
        gam1 *= 1e-5
        gam5 *= 1e-4
        p0 = p0max/(1+gam1*y[1])
        eta2 = eta2maxbar/(1+gam5*y[0])

        dydt[0] = (2*p0-1)*eta1*y[0]
        dydt[1] = 2*(1-p0)*eta1*y[0] - eta2*y[1]
        return dydt

    def _solve_ode(self, y0):
        '''The function solves func ode given IC y0, t, and model parameters
        theta'''
        y_soln = odeint(func=self.toy_lineage_ode, y0=y0, t=self.data_dict['t'],
                        args=(self.data_dict['theta'],))
        return y_soln

    def artificial_data(self, seed=None):
        '''Create [len(self.days_data)*self.data_dict['dpts'],2] vector with mean at
            the IC and SD at sqrt(sigmat^2+sigmab^2)'''
        if not seed:
            random.seed(self.data_dict['seed'])
        self.y_hat = self._solve_ode(self.data_dict['y0'])
        self.y_hat_used = np.zeros((self.data_dict['dpts']*self.N_time_points,2))

        size_rvs = self.N_time_points*self.data_dict['dpts']
        if self.log_data:
            scale_e = sqrt(self.data_dict['sigmate']**2 + self.data_dict['sigmabe']**2)
            self.y_ic = random.normal(loc=np.log(self.data_dict['y0']),
                                      scale=[scale_e, scale_e],
                                      size=(size_rvs, 2))
            self.y_ic = exp(self.y_ic)
        else:
            scale = sqrt(self.data_dict['sigmat']**2 + self.data_dict['sigmab']**2)
            self.y_ic = random.normal(loc=self.data_dict['y0'],
                                      scale=[scale, scale],
                                      size=(size_rvs, 2))

        #Initialize the artificial data vector
        TOT_RECORDS = self.data_dict['dpts']*self.N_time_points
        self.y_noisy_time_series = np.zeros((TOT_RECORDS,2))

        #First self.data_dict['dpts'] elements are the first self.data_dict['dpts'] elements from y_ic
        self.y_noisy_time_series[:self.data_dict['dpts'],:] = (
            self.y_ic[:self.data_dict['dpts']]
        )
        self.y_hat_used[:self.data_dict['dpts'],:] = self.data_dict['y0']

        '''Solve ODE for remaining y_ic (y_ic[:end]), add aditional technical
           noise, and add to artifical data vector y_noisy_time_series'''
        self.ixs = np.sort(np.tile(self.days_data_idx[1:], self.data_dict['dpts']))
        # print("yhat indexes: ", self.ixs)
        for i,j in enumerate(self.y_ic[self.data_dict['dpts']:]):
            y = self._solve_ode(j)
            if self.log_data:
                self.y_noisy_time_series[self.data_dict['dpts']+i,:] = (
                    exp(random.normal(loc=log(y[self.ixs[i], :]),
                          scale=[self.data_dict['sigmate'],self.data_dict['sigmate']]))
                    )
            else:
                self.y_noisy_time_series[self.data_dict['dpts']+i,:] = (
                    random.normal(loc=y[self.ixs[i], :],
                        scale=[self.data_dict['sigmat'],self.data_dict['sigmat']])
                    )
            self.y_hat_used[self.data_dict['dpts']+i,:] = y[self.ixs[i], :]
        return self.y_noisy_time_series

    def get_y_hat_true_df(self):
        y_hat_df = (
            np.concatenate(((self.data_dict['t']
                            ).reshape(self.data_dict['t'].shape[0], 1),
                             np.array(self.y_hat),
                             np.zeros([self.y_hat.shape[0],1])), axis=1)
        )
        cols = ["time", "HSC", "MPP", "used"]
        self.y_hat_df = pd.DataFrame(y_hat_df, columns=cols)
        self.y_hat_df.iloc[self.days_data_idx, -1] = 1
        self.y_hat_df.name = "True_ODE_solution"
        return self.y_hat_df
