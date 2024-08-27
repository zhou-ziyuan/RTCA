import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Generate some data
x = [0,1,2]
# y_fgsm = [197.37, 0.9, 0.42]
# y_atla = [197.37, 0.97, 1]
# y_paad = [197.37, 0.77, 0.52]
# y_rtcs = [197.37, 0.84, 0.29]


# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18, 3))


# x = [0,1,2]
y_rand = [197.37, 202.22, 199.59]
y_fgsm = [197.37, 198.28, 190.40]
y_pgd = [197.37, 195.15, 193.13]
y_sgld =  [197.37, 199.09, 195.56]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [197.37, 195.95, 194.44]
y_paad = [197.37, 192.12, 192.93]
y_rtcs1 = [197.37, 191.92, 192.63]
y_rtcs2 = [197.37, 191.11, 188.89]
y_rtcs3 = [197.37, 183.43, 184.04]
y_rtcs4 = [197.37, 182.42, 183.33]
# Plot data on subplots
axs[3].set_xticks([0, 1, 2])
axs[3].plot(x, y_rand, label='RN', marker='o')
axs[3].plot(x, y_fgsm, label='FGSM', marker='o')
axs[3].plot(x, y_pgd, label='PGD', marker='o')
axs[3].plot(x, y_sgld, label='SGLD', marker='o')
axs[3].plot(x, y_atla, label='ATLA', marker='o')
axs[3].plot(x, y_paad, label='PAAD', marker='o')
axs[3].plot(x, y_rtcs1, label='AMCA_GP', marker='o',  linestyle='--')
axs[3].plot(x, y_rtcs2, label='AMCA_GS', marker='o',  linestyle='--')
axs[3].plot(x, y_rtcs3, label='AMCA_DP', marker='o',  linestyle='--')
axs[3].plot(x, y_rtcs4, label='AMCA_DS', marker='o',  linestyle='--')
# axs[0].set_ylabel('WR')
axs[3].set_title('3a')
# axs[0].legend(loc='upper left')

# x = [0,1,2]
y_rand = [338.88, 332.82, 327.07]
y_fgsm = [338.88, 324.14, 321.82]
y_pgd = [338.88, 327.07, 320.30]
y_sgld =  [338.88, 325.55, 318.28]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [338.88, 327.67, 321.01]
y_paad = [338.88, 320.61, 312.93]
y_rtcs1 = [338.88, 328.78, 322.32]
y_rtcs2 = [338.88, 328.79, 322.52]
y_rtcs3 = [338.88, 279.79, 294.64]
y_rtcs4 = [338.88, 308.78, 289.09]
axs[4].set_xticks([0, 1, 2])
axs[4].plot(x, y_rand, label='RN', marker='o')
axs[4].plot(x, y_fgsm, label='FGSM', marker='o')
axs[4].plot(x, y_pgd, label='PGD', marker='o')
axs[4].plot(x, y_sgld, label='SGLD', marker='o')
axs[4].plot(x, y_atla, label='ATLA', marker='o')
axs[4].plot(x, y_paad, label='PAAD', marker='o')
axs[4].plot(x, y_rtcs1, label='AMCA_GP', marker='o',  linestyle='--')
axs[4].plot(x, y_rtcs2, label='AMCA_GS', marker='o',  linestyle='--')
axs[4].plot(x, y_rtcs3, label='AMCA_DP', marker='o',  linestyle='--')
axs[4].plot(x, y_rtcs4, label='AMCA_DS', marker='o',  linestyle='--')
# axs[1].set_ylabel('WR')
axs[4].set_title('6a')
# axs[1].legend()

y_rand = [574.85, 570.64, 555.48]
# y_fgsm = [574.85, 540.57, 535.02]
y_fgsm = [574.85, 540.57, 527.91]
# y_pgd = [574.85, 542.32, 533.17]
y_pgd = [574.85, 542.32, 519.15]
y_sgld =  [574.85, 546.64, 469.91]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [574.85, 581.38, 570.85]
y_paad = [574.85, 488.26, 371.51]
y_rtcs1 = [574.85, 445.94, 359.97]
y_rtcs2 = [574.85, 551.98, 508.52]
y_rtcs3 = [574.85, 416.53, 407.08]
y_rtcs4 = [574.85, 553.54, 488.53]
axs[1].set_xticks([0, 1, 2])
axs[1].plot(x, y_rand, label='RN', marker='o')
axs[1].plot(x, y_fgsm, label='FGSM', marker='o')
axs[1].plot(x, y_pgd, label='PGD', marker='o')
axs[1].plot(x, y_sgld, label='SGLD', marker='o')
axs[1].plot(x, y_atla, label='ATLA', marker='o')
axs[1].plot(x, y_paad, label='PAAD', marker='o')
axs[1].plot(x, y_rtcs1, label='AMCA_GP', marker='o',  linestyle='--')
axs[1].plot(x, y_rtcs2, label='AMCA_GS', marker='o',  linestyle='--')
axs[1].plot(x, y_rtcs3, label='AMCA_DP', marker='o',  linestyle='--')
axs[1].plot(x, y_rtcs4, label='AMCA_DS', marker='o',  linestyle='--')
# axs[2].set_ylabel('WR')
axs[1].set_title('Humanoid')
# axs[2].legend()

y_rand = [573.33, 743.56, 750.97]
# y_fgsm = [573.33, 820.19, 806.54]
y_fgsm = [573.33, 786.78, 711.74]
# y_pgd = [573.33, 652.52, 569.20]
y_pgd = [573.33, 777.71, 602.14]
y_sgld =  [573.33, 558.91, 496.52]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [573.33, 541.52, 516.79]
y_paad = [573.33, 799.58, 875.86]
y_rtcs1 = [573.33, 529.03, 494.93]
y_rtcs2 = [573.33, 565.74, 666.20]
y_rtcs3 = [573.33, 463.01, 379.87]
y_rtcs4 = [573.33, 507.94, 577.66]

axs[2].set_xticks([0, 1, 2])
axs[2].plot(x, y_rand, label='RN', marker='o')
axs[2].plot(x, y_fgsm, label='FGSM', marker='o')
axs[2].plot(x, y_pgd, label='PGD', marker='o')
axs[2].plot(x, y_sgld, label='SGLD', marker='o')
axs[2].plot(x, y_atla, label='ATLA', marker='o')
axs[2].plot(x, y_paad, label='PAAD', marker='o')
axs[2].plot(x, y_rtcs1, label='AMCA_GP', marker='o',  linestyle='--')
axs[2].plot(x, y_rtcs2, label='AMCA_GS', marker='o',  linestyle='--')
axs[2].plot(x, y_rtcs3, label='AMCA_DP', marker='o',  linestyle='--')
axs[2].plot(x, y_rtcs4, label='AMCA_DS', marker='o',  linestyle='--')
# axs[3].set_ylabel('WR')
axs[2].set_title('Ant')
# axs[3].legend()

y_rand = [111.31, 103.42, 108.98]
y_fgsm = [111.31, 102.87, 85.52]
y_pgd = [111.31, 92.13, 81.07]
y_sgld =  [111.31, 96.72, 84.41]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [111.31, 111.55, 99.48]
y_paad = [111.31, 100.68, 70.83]
y_rtcs1 = [111.31, 100.51, 89.80]
y_rtcs2 = [111.31, 102.39, 103.92]
y_rtcs3 = [111.31, 52.91, 57.95]
y_rtcs4 = [111.31, 61.89, 72.18]

axs[0].set_xticks([0, 1, 2])
axs[0].plot(x, y_rand, label='RN', marker='o')
axs[0].plot(x, y_fgsm, label='FGSM', marker='o')
axs[0].plot(x, y_pgd, label='PGD', marker='o')
axs[0].plot(x, y_sgld, label='SGLD', marker='o')
axs[0].plot(x, y_atla, label='ATLA', marker='o')
axs[0].plot(x, y_paad, label='PAAD', marker='o')
axs[0].plot(x, y_rtcs1, label='AMCA_GP', marker='o',  linestyle='--')
axs[0].plot(x, y_rtcs2, label='AMCA_GS', marker='o',  linestyle='--')
axs[0].plot(x, y_rtcs3, label='AMCA_DP', marker='o',  linestyle='--')
axs[0].plot(x, y_rtcs4, label='AMCA_DS', marker='o',  linestyle='--')
# axs[3].set_ylabel('WR')
axs[0].set_title('CUME')

# Display the figure
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.legend(loc='lower left', bbox_to_anchor=(0, 1))
plt.savefig('p_facmac.png', format='png')
plt.savefig('p_facmac.eps', format='eps')