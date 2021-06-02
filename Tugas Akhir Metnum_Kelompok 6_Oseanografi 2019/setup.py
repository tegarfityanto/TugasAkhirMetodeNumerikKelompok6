import os
os.system('color a')
setting = int(input("Apakah library Numpy, IPython, Matplotlib, Math, Sys, Scipy.linalg, sudah anda install? 1: Ya, 2: Tidak>>"))
if setting == 1:
 os.system('pip install numpy')
 os.system('pip install matplotlib')
 os.system('pip install ipython')
else:
 print("Baik kita lanjutkan!")
os.system('python code/codemetnumX.py')
k=input("Tekan ENTER untuk exit")