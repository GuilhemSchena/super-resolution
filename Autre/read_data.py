import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Récupérer les données

dfsrcnn = pd.read_csv(r"C:\Users\guilh\OneDrive\Documents\Travail\ETS\SYS843\Projet\2_Expérimentation\resultats\SRCNN\x2_t91g100aug\result_x2_lr0.0001_batch16_epoch300.csv", delimiter = ';')
epoch_srcnn = dfsrcnn. iloc[:, 0]
erreur_srcnn = dfsrcnn. iloc[:, 1]
psnr_srcnn = dfsrcnn. iloc[:, 2]

dffsrcnn = pd.read_csv(r"C:\Users\guilh\OneDrive\Documents\Travail\ETS\SYS843\Projet\2_Expérimentation\resultats\FSRCNN\fsrcnn32_5_1result_x2_lr0.001_batch16_epoch300.csv", delimiter = ';')
epoch_fsrcnn = dffsrcnn. iloc[:, 0]
erreur_fsrcnn = dffsrcnn. iloc[:, 1]
psnr_fsrcnn = dffsrcnn. iloc[:, 2]

dfdrcn = pd.read_csv(r"C:\Users\guilh\OneDrive\Documents\Travail\ETS\SYS843\Projet\2_Expérimentation\resultats\DRCN\D9F96nb300\result_DRCN_x2_nbepochs300_endlr1e-07_batch41_featurenum96_inferencedepth9.csv", delimiter = ';')
epoch_drcn = dfdrcn. iloc[::7, 0]
epoch_drcn = epoch_drcn.drop(epoch_drcn.index[-15])
erreur_drcn = dfdrcn. iloc[::7, 1]
psnr_drcn = dfdrcn. iloc[::7, 2]
psnr_drcn = psnr_drcn.iloc[5:304]

dfsrgan = pd.read_csv(r"C:\Users\guilh\OneDrive\Documents\Travail\ETS\SYS843\Projet\2_Expérimentation\resultats\SRGAN\srf_2_train_results.csv", delimiter = ';')
epoch_srgan = dfsrgan. iloc[:, 0]
erreur_srgan = dfsrgan. iloc[:, 1]
psnr_srgan = dfsrgan. iloc[0:299, 5]


### PLotting

# x axis values
x = list(map(int, epoch))
# corresponding y axis values
y_err_srcnn = list(map(float, erreur_srcnn))
y_psnr_srcnn = list(map(float, psnr_srcnn))

y_err_fsrcnn = list(map(float, erreur_fsrcnn))
y_psnr_fsrcnn = list(map(float, psnr_fsrcnn))

y_err_drcn = list(map(float, erreur_drcn))
y_psnr_drcn = list(map(float, psnr_drcn))

y_err_srgan = list(map(float, erreur_srgan))
y_psnr_srgan = list(map(float, psnr_srgan))

# plotting the points
plt.plot(x, psnr, label='srcnn')
# plt.plot(x, y_psnr_fsrcnn, label='fsrcnn')
# plt.plot(x, y_psnr_drcn, label='drcn')

plt.plot(x, y_psnr_srgan, label='srgan')


# naming the x axis
plt.xlabel('epoch')
# naming the y axis
plt.ylabel('PSNR (dB)')

# giving a title to my graph
plt.legend()

# function to show the plot
plt.show()