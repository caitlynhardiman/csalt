import matplotlib.pyplot as plt
from csalt.models import * 

def plot(data_, fixed_, code_, theta, mcube=None):
    # loop through observations
    
    for EB in range(data_['nobs']):

        # get the inference dataset
        dat = data_[str(EB)]

        # calculate model visibilities
        if mcube is None:
            mvis, mcube = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)], code=code_)
        else:
            mvis, mcube = vismodel_iter(theta, fixed_, dat,
                             data_['gcf'+str(EB)], data_['corr'+str(EB)], code=code_,
                             mcube=mcube)

        # spectrally bin the model
        wt = dat.iwgt.reshape((dat.npol, -1, dat.chbin, dat.nvis))
        mvis_b = np.average(mvis.reshape((dat.npol, -1, dat.chbin,
                                          dat.nvis)), weights=wt, axis=2)
        
        # plot for comparison
        deprojected_baseline = []
        for idx in range(len(dat.um)):
            coord = np.sqrt(dat.um[idx]**2 + dat.vm[idx]**2)
            deprojected_baseline.append(coord)

        plt.figure()
        plt.scatter(deprojected_baseline, np.real(dat.vis[0][150]), label='data')
        plt.scatter(deprojected_baseline, np.real(mvis_b[0][150]), label='model')
        plt.xlabel('Deprojected Baseline')
        plt.ylabel('Visibility')
        plt.legend()
        imgname = 'residplots/residuals'+str(EB)+'.png'
        plt.savefig(imgname)
        #plt.show()
        plt.close()
      
        return mcube
