import numpy as np
import pylab as pl
import matplotlib.animation as animation
import h5py
import imageio


list_cube_fake = []
with h5py.File('./Samples_ankit/Fake_Good_0.h5', 'r') as fh5:
    cube_fake = np.array(fh5['data'])
    list_cube_fake.append(cube_fake)

list_cube_real = []
with h5py.File('./Samples_ankit/Real_0.h5', 'r') as fh5:
    cube_real = np.array(fh5['data'])
    list_cube_real.append(cube_real)

volume1 = list_cube_real[0]
volume2 = list_cube_fake[0]

with imageio.get_writer('movie_black.gif', mode='I') as writer:
    list_img=[]
    for di in range(32):

        pl.figure(figsize=(14,8))
        pl.subplot(1,2,1)
        pl.imshow(volume1[di], interpolation='nearest', cmap=pl.cm.plasma)
        pl.clim(-1.0, 3.5)
        pl.xticks([]); pl.yticks([])

        pl.subplot(1,2,2)
        pl.imshow(volume2[di], interpolation='nearest', cmap=pl.cm.plasma)
        pl.clim(-1.0, 3.5)
        pl.xticks([]); pl.yticks([])

        pl.gcf().set_facecolor('black')

        # pl.show()

        fname='image_{:d}.png'.format(di)
        pl.savefig(fname, facecolor=pl.gcf().get_facecolor())
        print 'saved {:s}'.format(fname)
        image = imageio.imread(fname)
        list_img.append(image)
        writer.append_data(image)


