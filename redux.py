
import numpy as np
from astropy.io import fits, ascii
import glob
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
from astropy.table import Table

def fits_to_img(files):
    headers, datas, objs, dates, exps = np.empty(len(files), dtype=object), np.empty(len(files), dtype=object), \
                               np.empty(len(files), dtype=object), np.empty(len(files), dtype=object), \
                                        np.empty(len(files), dtype=object)
    for idx, file in enumerate(files):
        hdu_list = fits.open(file, memmap=True)
        data = hdu_list[0].data
        head = hdu_list[0].header
        obj = head['OBJECT']
        date = head['DATE-OBS'][:10]
        exp = head['EXPTIME']
        headers[idx] = head
        datas[idx] = np.array(data)
        objs[idx] = obj
        dates[idx] = date
        exps[idx] = exp
    print('Finished extracting {} file(s)'.format(obj))
    return headers, datas, objs, dates, exps

def make_bias(data):
    stack = np.stack(data)
    med = np.median(stack, axis=0) #take median along shortest axis
    return med

def make_flat(data, exps, bias_frame):
    data_stack = np.stack(data)
    bias_subbed = data_stack - bias_frame
    for i, e in enumerate(exps):
        bias_subbed[i] = bias_subbed[i]/e
    med = np.nanmedian(bias_subbed, axis=0)
    return med

def tophat(sig):
    topht = np.repeat(np.min(sig), len(sig))
    topht[sig > np.max(sig) / 2] = np.max(sig)
    return topht

def scatter_light_sub_and_order_trace(flat_frame):
    xx, yy = np.meshgrid(np.arange(0,flat_frame.shape[1]), np.arange(0,flat_frame.shape[0]))

    #this may need to change to be more flexible
    template = tophat(flat_frame[440:468, 0])
    #find and stores centers of each order
    order_centers = []
    columns = []
    for i in range(21):
        pix_column = 100*i
        flat_slice = flat_frame[:, pix_column]
        #template = tophat(flat_slice[440:468])
        corr = signal.correlate(flat_slice, template, mode='same')
        peaks, properties = signal.find_peaks(corr, distance=13, prominence=(5e6, 1e20))
        if i == 0:
            peaks = peaks[1:]
            num_of_ords = len(peaks)
        order_centers.append(peaks[:num_of_ords])
        columns.append(pix_column)
    scat_lights = []
    peak_s = []
    trough_s = []
    for pix_column in range(2048):
        flat_slice = flat_frame[:, pix_column]
        corr = signal.correlate(flat_slice, template, mode='same')
        peaks, properties = signal.find_peaks(corr, distance=13, prominence=(5e6, 1e20))
        troughs, props = signal.find_peaks(-flat_slice, distance=13, prominence=(5e2, 1e20))
        # order_centers.append(peaks[:num_of_ords])
        # columns.append(pix_column)
        troughs = np.append(np.arange(0,306, 1),troughs)
        scat_light = np.interp(np.arange(0,len(flat_slice), 1), troughs, flat_slice[troughs])
        scat_lights.append(scat_light)
        peak_s.append(peaks)
        trough_s.append(troughs)
        #plot one row to exemplify scattered light subtraction:
        if pix_column == 1000: #1310 bad
            fig, axs = plt.subplots(2,1,figsize=(8,8))
            ax1 = axs[0]
            ax2 = axs[1]
            ax1.plot(flat_slice, color='gray')
            ax1.plot(scat_light)
            ax1.plot(peaks, flat_slice[peaks], "x", color="blue")
            ax1.plot(troughs, flat_slice[troughs], "x", color="red")
            ax1.set_title("Scattered Light Subtraction (top panel = before, bottom = after)")
            ax2.plot(flat_slice - scat_light, color='gray')
            ax2.plot(peaks, flat_slice[peaks] - scat_light[peaks], "x", color="blue")
            ax2.plot(troughs, flat_slice[troughs] - scat_light[troughs], "x", color="red")
            plt.show()
        flat_frame[:, pix_column] = flat_slice - scat_light
        print("Scattered Light Subtraction: ", int(100*pix_column/2048), ' %', end='\r' )
    order_centers = np.array(order_centers).T
    columns = np.array(columns)
    print('\n')
    column_grid = np.tile(columns, order_centers.shape[0]).reshape(order_centers.shape)
    orders = []
    for i in range(column_grid.shape[0]):
        print("Order Tracing: ", int(100*i/column_grid.shape[0]), ' %', end='\r' )
        p = np.polynomial.Chebyshev.fit(column_grid[i], order_centers[i], 9)
        order = np.where((yy<p(xx[i])+5) & (yy>p(xx[i])-5))
        orders.append(order)

    normed = flat_frame/np.nanmax(flat_frame)
    normed[np.where(normed == 0)] = np.nan

    print('\n Scattered light subtraction and order tracing complete.')
    return normed, columns, order_centers, num_of_ords, orders, xx, yy


print("Redux.py successfully loaded in.")
#print(np.nanmin(flat_frame_true), np.nanmax(flat_frame_true), np.nanmedian(flat_frame_true))
#exit()
# plt.figure(figsize=(10,10))
# plt.imshow(flat_frame_true, cmap='gray', vmin=0.1, vmax=1)
# plt.scatter(column_grid[0], order_centers[0], marker='.', s=5, color='red')
# plt.scatter(xx[orders[0]], yy[orders[0]], s=8, color='orange', marker='+')
# plt.scatter(xx[orders[10]], yy[orders[10]], s=8, color='red', marker='+')
# plt.scatter(xx[orders[54]], yy[orders[54]], s=8, color='purple', marker='+')
# plt.show()
# plt.close()
# exit()
#
# plt.figure(figsize=(10,10))
# #plt.scatter(xx[orders[10]], yy[orders[10]], s=8, color='red', marker='+')
# plt.scatter(xx[orders[10]], flat_frame[orders[10]], s=8, color='red', marker='+')
# #plt.ylim(690, 540)
# plt.show()
# plt.close()



# plt.figure(figsize=(10,10))
# #plt.scatter(xx[orders[10]], yy[orders[10]], s=8, color='red', marker='+')
# plt.plot(waves[11], fluxes[11], color='red')
# #plt.ylim(690, 540)
# plt.show()
# plt.close()
#exit()

def extract_obj_spec(data, obj, exp, bias_frame, flat_frame, orders, num_of_ords, stack_and_take_mean=False):
    print("Extracting 1d spectrum of ", obj)
    xx, yy = np.meshgrid(np.arange(0,flat_frame.shape[1]), np.arange(0,flat_frame.shape[0]))
    if stack_and_take_mean == True:
        data = np.nansum(np.stack(data), axis=0)/len(data)
    data = data - bias_frame
    data = data/exp
    data = data/flat_frame
    waves = []
    fluxes = []
    for ord in range(num_of_ords):
        print("Spectral Extraction: ", int(100*ord/num_of_ords), ' %', end='\r' )
        wavs = np.array(list(set(xx[orders[ord]])))
        fluxi = np.zeros(len(wavs))
        for ind, w in enumerate(wavs):
            spot = np.where(xx[orders[ord]] == w)
            flux = np.nansum(data[orders[ord]][spot])
            fluxi[ind] = flux
        #waves.append(np.flip(wavs[wavs<2049])+(3000*ord))
        fluxes.append(fluxi[wavs<2049])
        waves.append(np.flip(wavs[wavs<2049]))
    waves = np.array(waves)
    fluxes = np.array(fluxes)
    print("Finished extracting 1d spectrum of ", obj, '\n')
    return waves, fluxes
print("\n")
#pixels, fluxes = extract_obj_spec(obj_data[0], obj_exps[0], bias_frame, flat_frame_true)

#thar_data_mean = np.nansum(thar_data, axis=0)


#thar_pixels, thar_fluxes = extract_obj_spec(thar_data, thar_exps[0], bias_frame, flat_frame_true, thar=True)

#set initial guess of dispersion per pixel
# R = 60000 #resolution guess
# pix_dispersion = 6500/R #dispersion around middle of spectrum given resolution guesses
#
# sun_pixels_stretched = sun_pixels * pix_dispersion
#
# sun_waves = []
# sun_ints = []
# sun_ords = []
# for ord in range(num_of_ords):
#     #peaks, properties = signal.find_peaks(thar_fluxes[ord], distance=1, prominence=(40, 10000))
#     sun_waves.append(sun_pixels[ord])
#     sun_ints.append(sun_fluxes[ord])
#     sun_ords.append(np.repeat(ord, len(sun_pixels[ord])))
#     # if ord in [5, 10, 15, 20, 25, 45]:
#     #     print(ord, peaks, properties)
#     #     plt.figure()
#     #     plt.plot(thar_pixels[ord], thar_fluxes[ord])
#     #     plt.scatter(thar_pixels[ord][peaks], thar_fluxes[ord][peaks], marker='x')
#     #     plt.show()
#     #     plt.close()
# sun_tab = Table([np.concatenate(sun_ords).ravel(), np.concatenate(sun_waves).ravel(), np.concatenate(sun_ints).ravel()], names=('Ord', 'Pix', 'Counts'))
# sun_tab.write("sun_obs.dat", format='ascii', overwrite=True)
# exit()

#ignore#
# thar_pixels_flat = list(np.ravel(np.array(thar_pixels)))
# thar_fluxes_flat = list(np.ravel(np.array(thar_fluxes)))
# sun_pixels_flat = list(np.ravel(np.array(sun_pixels)))
# sun_fluxes_flat = list(np.ravel(np.array(sun_fluxes)))


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    # print 'x = %d, y = %d'%(
    #     ix, iy)

    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))

    # Disconnect after 2 clicks
    # if len(coords) == 15:
    #     fig.canvas.mpl_disconnect(cid)
    #     plt.close(1)
    return

# thar_lines = Table.read("/Users/catherinemanea/Downloads/mnras0378-0221-SD1.txt", format='csv')
# # thar_lines.sort('ip')
# thar_w = np.linspace(np.min(thar_lines['wave']), np.max(thar_lines['wave']), 10000000)
# thar_f = np.zeros(len(thar_w))
# print(len(thar_lines))
# for i, w in enumerate(thar_lines['wave']):
#     print(i)
#     spot = np.where(np.abs(thar_w-w) < 1)
#     thar_f[spot] = thar_lines['Int'][i]
#
# t = Table([thar_w, thar_f], names=('w', 'f'))
# t.write("thar.dat", format='ascii')
# ta = Table.read("/Users/catherinemanea/Downloads/thar_spec_MM201006.dat", format='ascii')
# ta['col4'][np.where(ta['col4'] < 0)] = 0
# ta['col3'][np.where(ta['col3'] < 0)] = 0
# thar_w, thar_f = ta['col1'], ta['col3']
#
# thar_waves = []
# thar_ints = []
# thar_ords = []
# for ord in range(num_of_ords):
#     peaks, properties = signal.find_peaks(thar_fluxes[ord], distance=1, prominence=(40, 10000))
#     thar_waves.append(thar_pixels[ord])
#     thar_ints.append(thar_fluxes[ord])
#     thar_ords.append(np.repeat(ord, len(thar_pixels[ord])))
#     # if ord in [5, 10, 15, 20, 25, 45]:
#     #     print(ord, peaks, properties)
#     #     plt.figure()
#     #     plt.plot(thar_pixels[ord], thar_fluxes[ord])
#     #     plt.scatter(thar_pixels[ord][peaks], thar_fluxes[ord][peaks], marker='x')
#     #     plt.show()
#     #     plt.close()
# thar_tab = Table([np.concatenate(thar_ords).ravel(), np.concatenate(thar_waves).ravel(), np.concatenate(thar_ints).ravel()], names=('Ord', 'Pix', 'Prominence'))
# thar_tab.write("thar_obs.dat", format='ascii', overwrite=True)
# #
#


# from PyAstronomy import pyasl
# sol_wvl, sol_flx = pyasl.read1dFitsSpec("/Users/catherinemanea/Downloads/sun.ms.fits")

# fig = plt.figure()
#
# plt.plot(thar_pixels[20], thar_fluxes[20])
#
#
# # for ord in range(num_of_ords)[:45]:
# #     plt.plot(0.055*sun_pixels[45-ord] + (45-ord)*0.055*2290 + 3272.19, sun_fluxes[ord])
# #plt.scatter(10*thar_lines['wave'][-2:], np.repeat(17500, len(thar_lines['wave'][-2:])), marker='X', s=200)
#
# #coords = []
# # Call click func
# #cid = fig.canvas.mpl_connect('button_press_event', onclick)
# #plt.ylim(-500, 50000)
# plt.show()
# plt.close()
#print(str(ord)+str(coords[0][0])+ ' & ' +str(coords[1][0]))
