import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Generate some data
x = [0,1,2]
# y_fgsm = [0.97, 0.9, 0.42]

# y_atla = [0.97, 0.97, 1]
# y_paad = [0.97, 0.77, 0.52]
# y_rtcs = [0.97, 0.84, 0.29]


# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 3))
# x = [0,1,2]
y_rand = [103.13, 102.42, 102.92]
y_fgsm = [103.13, 99.09, 101.52]
y_pgd = [103.13, 98.28, 104.04]
y_sgld =  [103.13, 98.58, 99.19]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [103.13, 103.23, 103.13]
y_paad = [103.13, 96.97, 101.92]
y_rtcs1 = [103.13, 100.80, 98.68]
y_rtcs2 = [103.13, 101.41, 100.51]
y_rtcs3 = [103.13, 99.49, 96.26]
y_rtcs4 = [103.13, 100.30, 100.30]
# Plot data on subplots
axs[1].set_xticks([0, 1, 2])
axs[1].plot(x, y_rand, label='RN', marker='o')
axs[1].plot(x, y_fgsm, label='FGSM', marker='o')
axs[1].plot(x, y_pgd, label='PGD', marker='o')
axs[1].plot(x, y_sgld, label='SGLD', marker='o')
axs[1].plot(x, y_atla, label='ATLA', marker='o')
axs[1].plot(x, y_paad, label='PAAD', marker='o')
axs[1].plot(x, y_rtcs1, label='AMCA_GP', marker='o', linestyle='--')
axs[1].plot(x, y_rtcs2, label='AMCA_GS', marker='o', linestyle='--')
axs[1].plot(x, y_rtcs3, label='AMCA_DP', marker='o', linestyle='--')
axs[1].plot(x, y_rtcs4, label='AMCA_DS', marker='o', linestyle='--')
# axs[0].set_ylabel('WR')
axs[1].set_title('3a')

# x = [0,1,2]
y_rand = [36.87, 37.17, 39.69]
y_fgsm = [36.87, 36.46, 35.66]
y_pgd = [36.87, 35.65, 39.09]
y_sgld =  [36.87, 36.77, 33.83]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [36.87, 37.68, 31.01]
y_paad = [36.87, 34.14, 36.77]
y_rtcs1 = [36.87, 38.08, 38.08]
y_rtcs2 = [36.87, 40.10, 37.47]
y_rtcs3 = [36.87, 30.00, 33.33]
y_rtcs4 = [36.87, 29.09, 32.42]
axs[2].set_xticks([0, 1, 3])
axs[2].plot(x, y_rand, label='RN', marker='o')
axs[2].plot(x, y_fgsm, label='FGSM', marker='o')
axs[2].plot(x, y_pgd, label='PGD', marker='o')
axs[2].plot(x, y_sgld, label='SGLD', marker='o')
axs[2].plot(x, y_atla, label='ATLA', marker='o')
axs[2].plot(x, y_paad, label='PAAD', marker='o')
axs[2].plot(x, y_rtcs1, label='AMCA_GP', marker='o', linestyle='--')
axs[2].plot(x, y_rtcs2, label='AMCA_GS', marker='o', linestyle='--')
axs[2].plot(x, y_rtcs3, label='AMCA_DP', marker='o', linestyle='--')
axs[2].plot(x, y_rtcs4, label='AMCA_DS', marker='o', linestyle='--')
# axs[1].set_ylabel('WR')
axs[2].set_title('6a')
# axs[1].legend()

y_rand = [124.85, 112.63, 110.68]
y_fgsm = [124.85, 115.88, 99.67]
y_pgd = [124.85, 92.01, 92.62]
y_sgld =  [124.85, 107.04, 91.19]
# y_ow = [0.97, 0.97, 0.42]
y_atla = [124.85, 120.05, 96.37]
y_paad = [124.85, 103.63, 48.33]
y_rtcs1 = [124.85, 101.58, 59.27]
y_rtcs2 = [124.85, 98.07, 81.95]
y_rtcs3 = [124.85, 54.54, 52.52]
y_rtcs4 = [124.85, 93.48, 79.54]
axs[0].set_xticks([0, 1, 1])

axs[0].plot(x, y_rand, label='RN', marker='o')
axs[0].plot(x, y_fgsm, label='FGSM', marker='o')
axs[0].plot(x, y_pgd, label='PGD', marker='o')
axs[0].plot(x, y_sgld, label='SGLD', marker='o')
axs[0].plot(x, y_atla, label='ATLA', marker='o')
axs[0].plot(x, y_paad, label='PAAD', marker='o')
axs[0].plot(x, y_rtcs1, label='AMCA_GP', marker='o',  linestyle='--')
axs[0].plot(x, y_rtcs2, label='AMCA_GS', marker='o',  linestyle='--')
axs[0].plot(x, y_rtcs3, label='AMCA_DP', marker='o', linestyle='--')
axs[0].plot(x, y_rtcs4, label='AMCA_DS', marker='o', linestyle='--')
# axs[1].set_ylabel('WR')
axs[0].set_title('CUME')

# y_rand = [0.84, 0.74, 0.87]
# y_fgsm = [0.84, 0.42, 0.03]
# y_pgd =  [0.84, 0.23, 0.00]
# y_ow = [0.84, 0.32, 0.00]
# y_atla = [0.84, 0.77, 0.81]
# y_paad = [0.84, 0.52, 0.03]
# y_rtcs = [0.84, 0.06, 0]
# axs[2].set_xticks([0, 1, 2])
# axs[2].plot(x, y_rand, label='RN', marker='o')
# axs[2].plot(x, y_fgsm, label='FGSM', marker='o')
# axs[2].plot(x, y_pgd, label='PGD', marker='o')
# # axs[2].plot(x, y_ow, label='OWFGSM', marker='o')
# axs[2].plot(x, y_atla, label='ATLA', marker='o')
# axs[2].plot(x, y_paad, label='PAAD', marker='o')
# axs[2].plot(x, y_rtcs, label='RTCA', marker='o')
# # axs[2].set_ylabel('WR')
# axs[2].set_title('3s5z')
# axs[2].legend()

# axs[3].legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Display the figure
plt.savefig('p_maddpg.png', format='png')
plt.savefig('p_maddpg.eps', format='eps')