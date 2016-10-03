import numpy as np

def pearson_corr_coef(x, y):
    """
    Calculate the Pearson correlation coefficient for x and y
    """
    x = np.array(x)
    y = np.array(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    cov = np.mean(x*y)-np.mean(x)*np.mean(y)
    return cov/(sig_x*sig_y)