import numpy as np
import emcee
import multiprocessing
from astropy.cosmology import FlatLambdaCDM
import cPickle as pickle
import sys

salt2_info_path = '/Users/samdixon/repos/IDRTools/salt2_info_idr.pkl'

z, mb_obs, c_obs, x1_obs, ce, x1e, mbe = pickle.load(open(salt2_info_path, 'rb'))

def short_log_likelihood(params):
    alpha = params[0]
    beta = params[1]
    MB = params[2]
    sigma_int = params[3]
    omega_m = params[4]

    if omega_m < 0:
        return -np.inf
    if sigma_int < 0:
        return -np.inf

    cosmo = FlatLambdaCDM(H0=70, Om0=omega_m)
    mu_cosmo = cosmo.distmod(z).value
    dmu = mb_obs - mu_cosmo - MB + alpha * x1_obs - beta * c_obs
    dmue = np.sqrt(mbe**2 + (alpha * x1e)**2 + (beta * ce)**2 + sigma_int**2)
    return -0.5*np.sum((dmu/dmue)**2) - 0.5*np.sum(np.log(dmue**2))


def negtwoLL(alpha, beta, cosmomu, Mabs,
             mobs, xobs, cobs,
             Vm, Vint, Vx, Vc, Vmx, Vmc, Vxc,
             meanx, meanc, Vpriorx, Vpriorc):
    """Returns -2*Log(Likelihood) for analytic marginalization over x1true and ctrue. Accepts scalars and vectors.
    Caution: integers present, so make sure everything is floating point going in!
    Caution: beta is negative!"""

    return meanc**2/Vpriorc + meanx**2/Vpriorx + (xobs*(-(cosmomu*Vc*Vmx) - Mabs*Vc*Vmx + mobs*Vc*Vmx - cobs*Vmc*Vmx + cobs*Vint*Vxc + cobs*Vm*Vxc + cosmomu*Vmc*Vxc + Mabs*Vmc*Vxc - mobs*Vmc*Vxc + (-(Vc*(Vint + Vm)) + Vmc**2)*xobs))/(Vmc**2*Vx + Vc*(Vmx**2 - (Vint + Vm)*Vx) - 2*Vmc*Vmx*Vxc + (Vint + Vm)*Vxc**2) - ((cosmomu + Mabs - mobs)*(cosmomu*Vc*Vx + Mabs*Vc*Vx - mobs*Vc*Vx + cobs*Vmc*Vx - cobs*Vmx*Vxc - cosmomu*Vxc**2 - Mabs*Vxc**2 + mobs*Vxc**2 + Vc*Vmx*xobs - Vmc*Vxc*xobs))/(Vmc**2*Vx + Vc*(Vmx**2 - (Vint + Vm)*Vx) - 2*Vmc*Vmx*Vxc + (Vint + Vm)*Vxc**2) + (cobs*(cobs*(Vmx**2 - (Vint + Vm)*Vx) - (cosmomu + Mabs - mobs)*(Vmc*Vx - Vmx*Vxc) + (-(Vmc*Vmx) + (Vint + Vm)*Vxc)*xobs))/(Vmc**2*Vx + Vc*(Vmx**2 - (Vint + Vm)*Vx) - 2*Vmc*Vmx*Vxc + (Vint + Vm)*Vxc**2) - (meanx*(Vmc**2*Vx + Vc*(Vmx**2 - (Vint + Vm)*Vx) - 2*Vmc*Vmx*Vxc + (Vint + Vm)*Vxc**2) + Vpriorx*(cobs*(-(Vmc*(Vmx + alpha*Vx)) + (Vint + Vm + alpha*Vmx)*Vxc) + mobs*(Vc*(Vmx + alpha*Vx) - Vxc*(Vmc + alpha*Vxc)) + cosmomu*(-(Vc*(Vmx + alpha*Vx)) + Vxc*(Vmc + alpha*Vxc)) + Mabs*(-(Vc*(Vmx + alpha*Vx)) + Vxc*(Vmc + alpha*Vxc)) + (-(Vc*(Vint + Vm + alpha*Vmx)) + Vmc*(Vmc + alpha*Vxc))*xobs))**2/(Vpriorx*(Vmc**2*Vx + Vc*(Vmx**2 - (Vint + Vm)*Vx) - 2*Vmc*Vmx*Vxc + (Vint + Vm)*Vxc**2)*(Vmc**2*(Vpriorx + Vx) + Vc*(Vmx**2 - 2*alpha*Vmx*Vpriorx - alpha**2*Vpriorx*Vx - (Vint + Vm)*(Vpriorx + Vx)) - 2*Vmc*(Vmx - alpha*Vpriorx)*Vxc + (Vint + Vm + alpha**2*Vpriorx)*Vxc**2)) + (meanc*(Vmc**2*(Vpriorx + Vx) + Vc*(Vmx**2 - 2*alpha*Vmx*Vpriorx - alpha**2*Vpriorx*Vx - (Vint + Vm)*(Vpriorx + Vx)) - 2*Vmc*(Vmx - alpha*Vpriorx)*Vxc + (Vint + Vm + alpha**2*Vpriorx)*Vxc**2) + Vpriorc*(-(Mabs*Vmc*Vpriorx) + mobs*Vmc*Vpriorx - Mabs*Vmc*Vx + mobs*Vmc*Vx + cobs*(Vmx**2 - 2*alpha*Vmx*Vpriorx - alpha**2*Vpriorx*Vx - (Vint + Vm)*(Vpriorx + Vx)) + Mabs*Vmx*Vxc - mobs*Vmx*Vxc - alpha*Mabs*Vpriorx*Vxc + alpha*mobs*Vpriorx*Vxc + meanx*(Vmc*(Vmx + alpha*Vx) - (Vint + Vm + alpha*Vmx)*Vxc) - cosmomu*(Vmc*(Vpriorx + Vx) - Vmx*Vxc + alpha*Vpriorx*Vxc) - Vmc*Vmx*xobs + alpha*Vmc*Vpriorx*xobs + Vint*Vxc*xobs + Vm*Vxc*xobs + alpha**2*Vpriorx*Vxc*xobs + beta*(-(cobs*Vmc*Vpriorx) - cobs*Vmc*Vx + cobs*Vmx*Vxc - alpha*cobs*Vpriorx*Vxc + mobs*(Vc*(Vpriorx + Vx) - Vxc**2) + cosmomu*(-(Vc*(Vpriorx + Vx)) + Vxc**2) + Mabs*(-(Vc*(Vpriorx + Vx)) + Vxc**2) + meanx*(Vc*(Vmx + alpha*Vx) - Vxc*(Vmc + alpha*Vxc)) - Vc*Vmx*xobs + alpha*Vc*Vpriorx*xobs + Vmc*Vxc*xobs)))**2/(Vpriorc*(Vmc**2*(Vpriorx + Vx) + Vc*(Vmx**2 - 2*alpha*Vmx*Vpriorx - alpha**2*Vpriorx*Vx - (Vint + Vm)*(Vpriorx + Vx)) - 2*Vmc*(Vmx - alpha*Vpriorx)*Vxc + (Vint + Vm + alpha**2*Vpriorx)*Vxc**2)*(-(Vmx**2*Vpriorc) - Vmc**2*Vpriorx + Vint*Vpriorc*Vpriorx + Vm*Vpriorc*Vpriorx + 2*beta*Vmc*Vpriorc*Vpriorx + 2*alpha*Vmx*Vpriorc*Vpriorx - Vmc**2*Vx + Vint*Vpriorc*Vx + Vm*Vpriorc*Vx + 2*beta*Vmc*Vpriorc*Vx + alpha**2*Vpriorc*Vpriorx*Vx + Vc*(-Vmx**2 + 2*alpha*Vmx*Vpriorx + alpha**2*Vpriorx*Vx + (Vint + Vm)*(Vpriorx + Vx) + beta**2*Vpriorc*(Vpriorx + Vx)) + 2*(Vmc - beta*Vpriorc)*(Vmx - alpha*Vpriorx)*Vxc - (Vint + Vm + beta**2*Vpriorc + alpha**2*Vpriorx)*Vxc**2)) - np.log(4) - 2*np.log(np.pi) + np.log(Vpriorc) + np.log(Vpriorx) + np.log(-(Vmc**2*Vx) + Vc*(-Vmx**2 + (Vint + Vm)*Vx) + 2*Vmc*Vmx*Vxc - (Vint + Vm)*Vxc**2) + np.log((Vmc**2*(Vpriorx + Vx) + Vc*(Vmx**2 - 2*alpha*Vmx*Vpriorx - alpha**2*Vpriorx*Vx - (Vint + Vm)*(Vpriorx + Vx)) - 2*Vmc*(Vmx - alpha*Vpriorx)*Vxc + (Vint + Vm + alpha**2*Vpriorx)*Vxc**2)/(Vpriorx*(Vmc**2*Vx + Vc*(Vmx**2 - (Vint + Vm)*Vx) - 2*Vmc*Vmx*Vxc + (Vint + Vm)*Vxc**2))) + np.log(-((-(Vmx**2*Vpriorc) - Vmc**2*Vpriorx + Vint*Vpriorc*Vpriorx + Vm*Vpriorc*Vpriorx + 2*beta*Vmc*Vpriorc*Vpriorx + 2*alpha*Vmx*Vpriorc*Vpriorx - Vmc**2*Vx + Vint*Vpriorc*Vx + Vm*Vpriorc*Vx + 2*beta*Vmc*Vpriorc*Vx + alpha**2*Vpriorc*Vpriorx*Vx + Vc*(-Vmx**2 + 2*alpha*Vmx*Vpriorx + alpha**2*Vpriorx*Vx + (Vint + Vm)*(Vpriorx + Vx) + beta**2*Vpriorc*(Vpriorx + Vx)) + 2*(Vmc - beta*Vpriorc)*(Vmx - alpha*Vpriorx)*Vxc - (Vint + Vm + beta**2*Vpriorc + alpha**2*Vpriorx)*Vxc**2)/(Vpriorc*(Vmc**2*(Vpriorx + Vx) + Vc*(Vmx**2 - 2*alpha*Vmx*Vpriorx - alpha**2*Vpriorx*Vx - (Vint + Vm)*(Vpriorx + Vx)) - 2*Vmc*(Vmx - alpha*Vpriorx)*Vxc + (Vint + Vm + alpha**2*Vpriorx)*Vxc**2))))

def full_log_likelihood(params):
    alpha = params[0]
    beta = params[1]
    MB = params[2]
    sigma_int = params[3]
    omega_m = params[4]
    meanx = params[5]
    meanc = params[6]
    sigpriorx = params[7]
    sigpriorc = params[8]
    if (omega_m < 0.0) or (omega_m > 1.0):
        return -np.inf
    if sigma_int < 0:
        return -np.inf
    if sigpriorx < 0:
        return -np.inf
    if sigpriorc < 0:
        return -np.inf

    cosmo = FlatLambdaCDM(H0=70, Om0=omega_m)
    cosmomu = cosmo.distmod(z).value
    return np.sum(negtwoLL(alpha, beta, cosmomu, MB,
                           mb_obs, x1_obs, c_obs,
                           mbe**2, sigma_int**2, x1e**2, ce**2,
                           0.0, 0.0, 0.0,
                           meanx, meanc, sigpriorx**2, sigpriorc**2)) / (-2.0)


nwalkers = 1000
ndim = 9
randarr = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
guess = [0.12, -3.0, -19.0, 0.15, 0.3, 0.3, 0.0, 0.8, 0.1]
steps = [0.08, 2.0, 2.0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]
nsamples = 25
nburn = 50
keys = 'alpha beta MB sigma_int omega_m meanx meanc sigpriorx sigpriorc'.split()

# ndim = 5
# randarr = np.random.rand(ndim*nwalkers).reshape((nwalkers, ndim))
# guess = [0.12, -3, -19, 0.15, 0.3]
# steps = [0.08, 2.0, 2.0, 0.1, 0.2]
# nsamples = 250
# nburn = 50
# keys = 'alpha beta MB sigma_int omega_m'.split()

start = np.array(guess) + np.array(steps) * (randarr - 0.5)
sampler = emcee.EnsembleSampler(nwalkers, ndim, full_log_likelihood, threads=8)

for i, result in enumerate(sampler.sample(start, iterations=nburn)):
    pos, prob, state = result
    print("Burn in: {0:5.1%}".format(float(i) / nburn))
sampler.reset()

for i, result in enumerate(sampler.sample(pos, iterations=nsamples)):
    pos, prob, state = result
    print("Sampling: {0:5.1%}".format(float(i) / nsamples))

maxprob = np.argmax(sampler.flatlnprobability)
chain_dict = {}
chain_dict['lnprob'] = sampler.flatlnprobability
for i in range(ndim):
    chain_dict[keys[i]] = sampler.flatchain[:, i]
    print(keys[i], sampler.flatchain[:, i][maxprob], np.median(sampler.flatchain[:, i]), np.mean(sampler.flatchain[:, i]))

pickle.dump(chain_dict, open('emcee_hr_full_nsamp25_noburn.pkl', 'wb'))

