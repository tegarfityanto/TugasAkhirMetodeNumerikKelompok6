#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Metode setengah interval atau bisection merupakan metode yang mampu  untuk menyelesaikan persamaan non-linier 
dengan membagi dua nilai x1 dan x2. Dilakukan secara berulang-ulang sampai nilai x lebih kecil dari nilai toleransi. 
Perubahan fungsi mencakup sebuah interval dengan memilih x1 bawah dan x2 puncak taksiran akar.
'''

#Metode Setengah Interval
def Setengah_Interval (X1,X2):
  X1 = X1
  X2 = X2
  error = 1
  iterasi = 0
  while(error > 0.0001):
      iterasi +=1
      FXi = 0.0533*(X1**3)-5.5006*(X1**2)+140.4*(X1)-1020.1
      FXii = 0.0533*(X2**3)-5.5006*(X2**2)+140.4*(X2)-1020.1
      Xt = (X1 + X2)/2
      FXt = 0.0533*(Xt**3)-5.5006*(Xt**2)+140.4*(Xt)-1020.1
      if FXi * FXt > 0:
          X1 = Xt
      elif FXi * FXt < 0:
          X2 = Xt
      else:
          print("Akar Penyelesaian: ", Xt)
      if FXt < 0:
          error = FXt * (-1)
      else:
          error = FXt
      if iterasi > 100:
          print("Angka tak hingga")
          break
      print(iterasi, "|", FXi, "|", FXii, "|", Xt, "|", FXt, "|", error)
  print("Jumlah Iterasi: ", iterasi)
  print("Akar persamaan adalah: ", Xt)
  print("Toleransi Error: ", error)


'''Metode ini memiliki kesamaan dengan metode setengah interval. Kesamaannya terletak dalam diperlukan dua harga taksiran 
awal pada awal pengurungan akar persamaan. Namun, metode ini melakukan pencarian pendekatan akar persamaan selanjutnya 
setelah pendekatan akar saat ini ditemukan. Interpolasi linier dilakukan melalui dua titik pertama dengan garis interpolasi 
memotong sumbu x dan dititik perpotongan tersebut didapatkan pendekatan akar yang pertama lalu dilakukan evaluasi untuk 
diperoleh titik pada fungsi nonliner dan diinterpolasi kembali hingga mendapatkan akar persamaan eksak.
'''

#Metode Interpolasi Linier (MODUL 2)

def Interpolasi_Linier (X1):
  X1 = X1
  X2 = X1 + 1
  error = 1
  iterasi = 0
  while(error > 0.0001):
      iterasi +=1
      FX1 = 0.0533*(X1**3)-5.5006*(X1**2)+140.4*(X1)-1020.1
      FX2 = 0.0533*(X2**3)-5.5006*(X2**2)+140.4*(X2)-1020.1
      Xt = X2 - ((FX2/(FX2-FX1)))*(X2-X1)
      FXt = 0.0533*(Xt**3)-5.5006*(Xt**2)+140.4*(Xt)-1020.1
      if FXt*FX1 > 0:
          X2 = Xt
          FX2 = FXt
      else:
          X1 = Xt
          FX1 = FXt
      if FXt < 0:
          error = FXt * (-1)
      else:
          error = FXt
      if iterasi > 500:
          print("Angka tak hingga")
          break
      print(iterasi, "|", FX1, "|", FX2, "|", Xt, "|", FXt, "|", error)
  print("Jumlah Iterasi: ", iterasi)
  print("Akar persamaan adalah: ", Xt)
  print("Toleransi Error: ", error)


'''
Metode secant menggunakan alogaritma yang sama dengan Newton-Rhapson, yang membedakan adalah pendekatan turunan pertama f. 
Pendekatan f' menggunakan ungkapan beda hingga yang didasarkan pada taksiran akar sebelumnya (beda mundur) sehingga metode 
Secant tidak memerlukan dua taksiran awal yang harus mengurung akar persamaan.
'''

#Metode Secant (MODUL 2)

def Secant (X1):
  X1 = X1
  X2 = X1 - 1
  error = 1
  iterasi = 0
  while(error > 0.0001):
      iterasi +=1
      FX1 = 0.0533*(X1**3)-5.5006*(X1**2)+140.4*(X1)-1020.1
      FXmin = 0.0533*(X2**3)-5.5006*(X2**2)+140.4*(X2)-1020.1
      X3 = X1 - ((FX1)*(X1-(X2)))/((FX1)-(FXmin))
      FXplus = 0.0533*(X3**3)-5.5006*(X3**2)+140.4*(X3)-1020.1
      if FXplus < 0:
          error = FXplus * (-1)
      else:
          error = FXplus
      if error > 0.0001:
          X2 = X1
          X1 = X3
      else:
          print("Selesai")
      if iterasi > 500:
          print("Angka tak hingga")
          break
      print(iterasi, "|", FX1, "|", FXmin, "|", X3, "|", FXplus, "|", error)
  print("Jumlah Iterasi: ", iterasi)
  print("Akar persamaan adalah: ", X3)
  print("Toleransi Error: ", error)


'''
Metode Newton-Raphson adalah metode populer yang sederhana yang hanya memburuhkan satu tekaan awal. Metode ini memerlukan 
perhitungan turunan fungsi dalam setiap iterasi. Sifar metode Newton-Rapshon tersebut berkaitan dengan fakta bahwa hampiran 
berikutnya diperoleh dengan menarik garis singgung kurva pada titik yang mempunyai absis hampiran sebelumnya hingga memotong 
sumbu-x. Titik potong garis singgung tersebut dengan sumbu-x merupakan hampiran berikutnya dan dilanjut hingga memenuhi syarat
keakuratan yang ditentukan.
'''

#Metode Newton-Rhapson (MODUL 2)

def Newton_Rhapson (X1):
  X1 = X1
  iterasi = 0
  akar = 1
  while (akar > 0.0001):
      iterasi += 1
      Fxn = 0.0533*(X1**3)-5.5006*(X1**2)+140.4*(X1)-1020.1
      Fxxn = ((3*0.0533)*(X1**2))-((2*5.5006)*X1)+140.4
      xnp1 = X1 - (Fxn/Fxxn)
      fxnp1 = (xnp1**3)+(xnp1**2)-(3*xnp1)-3
      Ea = ((xnp1-X1)/xnp1)*100
      if Ea < 0.0001:
          X1 = xnp1
          akar = Ea*(-1)
      else:
          akar = xnp1
          print("Nilai akar adalah: ", akar)
          print("Nilai error adalah: ", Ea)
      if iterasi > 100:
          break
      print(iterasi, "|", X1, "|", xnp1, "|", akar, "|", Ea)
  print("Jumlah Iterasi: ", iterasi)
  print("Akar persamaan adalah: ", xnp1)
  print("Toleransi Error: ", akar)


'''
Metode iterasi menggunakan alogaritma sederhana dengan memilih nilai batas sebagai nilai awal yang kemudian diulang-ulang
hingga mencapai keadaan konvergen yang merepresentasikan solusi dari sistem persamaan. Banyaknya perhitungan dalam metode
iterasi tidak dapat diprediksi sehingga konvergensi suatu sistem persamaan tidak selalu terjamin. Namun demikian,
metode iterasi menggunakan ruang memori yang kecil karena hanya menyimpan koefisien-koefisien bukan nol.
'''

#Metode Iterasi (MODUL 2)

def Iterasi (X1):
  X1 = X1
  error = 1
  iterasi = 0
  while (error > 0.0001):
      iterasi +=1
      Fxn = 0.0533*(X1**3)-5.5006*(X1**2)+140.4*(X1)-1020.1
      X2 = ((5.5006*(X1**2)-140.4*(X1)+1020.1)/0.0533)**(0.333334)
      Ea = (((X2-X1)/(X2))*100)
      if Ea < error:
          X1 = X2
          if Ea > 0:
              error = Ea
          else:
              error = Ea*(-1)
      else:
          error = Ea
      if iterasi > 100:
          print("Angka tak hingga")
          break
      print(iterasi, "|", X1, "|", X2, "|", Ea, "|", error)
  print("Jumlah Iterasi: ", iterasi)
  print("Akar persamaan adalah: ", X2)
  print("Toleransi Error: ", error)



'''
Metode eliminasi gauss termasuk dalam metode penyelesaian persamaan linear dengan cara langsung.
Inti dari metode ini adalah membawa persamaan kedalam bentuk matriks dan menyederhanakan matriks menjadi bentuk segitiga atas.
Untuk metode gauss Jordan adalah metode yang dikembangkan dari metode eliminasi gauss.
Dimana tujuan kita membuat matriks identitas bukan lagi segitiga atas
sehingga tidak diperlukan lagi subtitusi balik untuk mencari nilai dari akar persamaan.
'''

'''
Berbeda dengan 2 metode sebelumnya yaitu eliminasi gauss dan gauss jordan yang menggunakan matriks untuk penyelesaiannya
metode gauss seidel menggunakan metode iterasi dalam penyelesaiannya.
Metode ini dikembangkan dari gagasan penyelesaian persamaan tak linear.
'''

#PERHITUNGAN METODE ELEMINASI GAUSS (MODUL 3)
#Eliminasi gauss termasuk dalam metode penyelesaian persamaan linear dengan cara langsung

#Diketahui
#5x1 + 10x2 + 3x3 + 8x4 = 4.886
#1x1 + 5x2 + 6x3 + 6x4 = 9.086
#2x1 + 3x2 + 5x3 + 4x4 = 20.086
#3x1 + 6x2 + 6x3 + 5x4 = 10.086

import numpy as np

def Cal_LU_pivot(D, g):
    A = np.array((D), dtype=float)
    f = np.array((g), dtype=float)
    n = len(f)
    for i in range(0, n - 1):  # Looping untuk kolom matriks

        if np.abs(A[i, i]) == 0:
            for k in range(i + 1, n):
                if np.abs(A[k, i]) > np.abs(A[i, i]):
                    A[[i, k]] = A[[k, i]]  # Tukar antara baris i dan k
                    f[[i, k]] = f[[k, i]]
                    break

        for j in range(i + 1, n):  # Ulangi baris di bawah diagonal untuk setiap kolom
            m = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m * A[i, :]
            f[j] = f[j] - m * f[i]
    return A, f


#PERHITUNGAN METODE ELEMINASI GAUSS JORDAN (MODUL 3)
#metode gauss Jordan memiliki tujuan untuk membuat matriks identitas bukan lagi segitiga atas

#Diketahui
#5x1 + 10x2 + 3x3 + 8x4 = 4.886
#1x1 + 5x2 + 6x3 + 6x4 = 9.086
#2x1 + 3x2 + 5x3 + 4x4 = 20.086
#3x1 + 6x2 + 6x3 + 5x4 = 10.086

import numpy as np
import sys

def GaussJordan(a,n):
    #Step1 ===> Looping untuk pengolahan metode Gauss Jordan
    print('==============Mulai Iterasi===============')
    for i in range(n):
        if a[i][i]==0:
            sys.exit('Dibagi dengan angka nol (proses tidak dapat dilanjutkan)')
        for j in range(n):
            if i !=j:
                ratio=a[j][i]/a[i][i]
                #print('posisi nol di:[',j,i,']', 'nilai rasio:',ratio)
                
                for k in range (n+1):
                    a[j,k]=a[j][k]-ratio*a[i][k]
                print(a)
                print(f'============================================')
    # Step 2 ====> Membuat semua variabel(x,y,z,...)==1
    ax=np.zeros((n,n+1))
    for i in range(n):
        for j in range(n+1):
            ax[i,j]=a[i][j]/a[i][i]
    print('===================Akhir Iterasi============')
    return ax


#PERHITUNGAN METODE ITERASI GAUSS SIEDEL (MODUL 3)
#Dengan metode iterasi Gauss-Seidel pembulatan dapat diperkecil
#karena dapat meneruskan iterasi sampai solusinya seteliti mungkin sesuai dengan batas sesatan yang diperbolehkan

import numpy as np 
from scipy.linalg import solve 

def GaussSeidel (A, b, x, n):
    L = np.tril(A)
    U = A -L
    for i in range(n):
        x=np.dot(np.linalg.inv(L), b - np.dot(U,x))
        print (f'Iterasi Ke-{str(i+1).zfill(3)}'),
        print (x)
    return x

#PERHITUNGAN METODE ITERASI GAUSS JACOBI (MODUL 3)
#Metode iterasi Jacobi, prinsipnya: merupakan metode iteratif
#yang melakuakn perbaharuan nilai x yang diperoleh tiap iterasi.

import numpy as np
from scipy.linalg import solve

def jacobi(A,b,x,n):
  D = np.diag(A)
  R = A-np.diagflat(D)
  for i in range(n):
    x = (b-np.dot(R,x))/D
    print (f'Iterasi Ke-{str(i+1).zfill(3)}'),
    print (x)
  return x



'''
Integrasi numerik adalah suatu metode aproksimasi
untuk memperoleh nilai integral pada suatu fungsisecara numerik.
Metode ini dapat digunakan pada fungsi-fungsi yang diintegralkan dengan metode analitik agak sulit.
'''

'''
Metode trapesium adalah metode yang paling sederhana dan sebenarnyauntuk menghitung
luas suatu trapesium yaitu segi empat dimana sisi-sisi yang berhadapanya sejajar.
Metode ini melakukan pendekatan integral numerik dengan persamaan polinomial orde satu.
'''

'''
Metode ini dapat dibedakan menjadi 2 yaitu metode simpson 1/3 dan metode simpson 3/8.
Metode simpson 1/3 menggunakan polinomial orde 2.
Pada simpson 1/3 hanya berlaku untuk jumlah pias genap.
Sedangkan metode simpson 3/8 menggunakan polinomial orde 3.
Pada metode simpson 3/8 memerlukan 4 titik dalam perhitungannya.
'''

#METODE TRAPESIUM BANYAK PIAS (MODUL 4)
#Semakin kecil pias yang digunakan, hasil akan semakin teliti.
#Metode ini dapat digunakan pada jumlah pias ganjil.

import math
import numpy as np
from matplotlib import pyplot as plt

def trapesium(f,a,b,N):
    x = np.linspace(a,b,N+1)
    y = f(x)
    y_right = y[1:] 
    y_left = y[:-1] 
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)
    return T

#Simpson 1/3 (MODUL 4)
#Pada metode simpson 1/3 menggunakan polinomial orde 2
#namun dapat menghasilkan ketelitian orde 3 dan hanya memerlukan 3 titik.

def simpson1per3(x0,xn,n):
    f1= 3*(x0**3)+4*(x0**2)+6

    f2= 3*(xn**3)+4*(xn**2)+6

    h = (xn - x0) / n
    
    integral = f1 + f2
    
    for i in range(1,n):
        k = x0 + i*h
        
        if i%2 == 0:
            integral = integral + 2 * f(k)
        else:
            integral = integral + 4 * f(k)
    
    integral = integral * h/3
    
    return integral
    
    
    
'''
Persamaan diferensial adalah persamaan matematika untuk fungsi satu variabel atau lebih,
yang menghubungkan nilai fungsi itu sendiri dan turunannya dalam berbagai orde.
Persamaan diferensial biasa merupakan persamaan diferensial yang hanya mempunyai 
satu peubah bebas yang biasanya disimboklan dengan x. contohnya y^'=x^2+y^2 peubah bebasnya adalah x
sedangkan peubah terkaitnya adalah y yang merupakan fungsi dari x atau ditulis sebagai y=g(x).
'''

'''
Metode euler merupakan metode satu langkah yang paling sederhana namun dinilai kurang teliti.
Metode euler disebut juga dengan metode orde pertama karena pencarian
dalam persamaannya hanya mengambil sampai orde pertama saja.
Metode ini diturunkan dari persamaan deret Taylor.
'''

'''
Metode heun digunakan untuk menyelesaikan berbagai persoalan matematika yang mempunyai masalah nilai awal.
Metode Heun menaksir nilai yang membutuhkan satu buah taksiran nilai sebelumnya.
Metode Heun juga merupakan salah satu peningkatan dari metode Euler.
'''
    

#METODE EULER (MODUL 5)
#Dengan menggunakan pendekatan nilai awal (x0,y0)
#maka nilai-nilai y berikutnya dapat diperoleh seperti yang ditulis dalam (y)

#=== Import Library ===#
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

def Euler (h,x0,xn, Nama):
  plt.style.use('seaborn-poster')
  ipy = get_ipython()
  if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

#==== Mendefinisikan parameter dan fungsi ===#
  x = np.arange(x0, xn + h, h) # Numerical grid
  y0 = float(input("Masukkan nilai y awal: ")) # Initial Condition
  G = 2*((x**3)) + 9*(x**2) + 3*(x) + 86
  f = lambda x, y: 2*((x**3)) + 9*(x**2) + 3*(x) + 86 # Persamaan Differensial Biasa

#=== Metode Euler Eksplisit ===#
  y = np.zeros(len(x))
  y[0] = y0

  for i in range(0, len(x) - 1):
      y[i + 1] = y[i] + h*f(x[i], y[i])
    
  print(y)

#=== Menghitung Error ===#
  Galat = G-y
  print (Galat)

  Nama = Nama

#=== Menampilkan Grafik ===#
  Judul = ("\n Grafik Pendekatan Persamaan Differensial Biasa Dengan Metode Euler \n\n %s \n" % (Nama))
  plt.figure(figsize = (10, 8))
  plt.plot(x, y, 'b:', label='Hasil Pendekatan')
  plt.plot(x, G, '-g', label='Hasil Analitik')
  plt.title(Judul) # Judul plot
  plt.xlabel('x') # Label sumbu x
  plt.ylabel('F(x)') # Label sumbu y
  plt.grid() # Menampilkan grid
  plt.legend(loc='lower right')
  plt.savefig('image\euler.png')

#METODE HEUN (MODUL 5)
#Metode Heun menaksir nilai y(tn+1)
#Metode Heun membutuhkan satu buah taksiran nilai sebelumnya yaitu y(tn).

#=== Import Library ===#
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

def Heun (h,x0,xn, Nama):
  plt.style.use('seaborn-poster')
  ipy = get_ipython()
  if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

#==== Mendefinisikan parameter dan fungsi ===#
  x = np.arange(x0, xn + h, h) # Numerical grid
  y0 = float(input("Masukkan nilai y awal: ")) # Initial Condition
  G = 2*((x**3)) + 9*(x**2) + 3*(x) + 86
  f = lambda x, y: 2*((x**3)) + 9*(x**2) + 3*(x) + 86 # Persamaan Differensial Biasa

#=== Metode Heun / Runge Kutta Orde 2 ===#
  y = np.zeros(len(x))

  y[0] = y0
  for i in range(0, len(x) - 1):
      k1 = f(x[i], y[i])
      k2 = f((x[i]+h), (y[i]+(h*k1)))
      
      y[i+1] = y[i]+(0.5*h*(k1+k2))

  print(y)
    
#=== Menghitung Error ===#
  Galat = G-y
  print (Galat)

#=== Menampilkan Grafik ===#
  Judul = ("\n Grafik Pendekatan Persamaan Differensial Biasa Dengan Metode Heun \n\n %s \n" % (Nama))
  plt.figure(figsize = (8, 8))
  plt.plot(x, y, 'b--', label='Hasil Pendekatan')
  plt.plot(x, G, '-g', label='Hasil Analitik')
  plt.title(Judul) # Judul plot
  plt.xlabel('x') # Label sumbu x
  plt.ylabel('F(x)') # Label sumbu y
  plt.grid() # Menampilkan grid
  plt.legend(loc='lower right')
  plt.savefig('image\heun.png')



print("Kode pengguanaan akar-akar persamaan:\n",
      "1. Metode Setengah Interval \n",
      "2. Metode Interpolasi Linier \n",
      "3. Metode Secant \n",
      "4. Metode Newton-Rapson \n",
      "5. Metode Iterasi\n",
      "Kode pengguanaan Sistem Persamaan Linier dan Matrix:\n",
      "6. Metode Gauss \n",
      "7. Metode Gauss Jordan \n",
      "8. Metode Gauss Seidel \n",
      "9. Metode Jacobi \n",
      "Kode pengguanaan Intergrasi Numerik:\n",
      "10. Metode Trapesium Banyak Bias \n",
      "11. Metode Simpson 1/3 \n",
      "Kode pengguanaan Persamaan Differensial Biasa :\n",
      "12. Metode Euler \n",
      "13. Metode Heun \n")
Setting =int(input('Masukkan kode penggunaan :'))
if (Setting == 1):
  X1 =float(input("masukkan nilai X1 setengah interval :"))
  X2 =float(input("masukkan nilai X2 setengah interval :"))
  X = Setengah_Interval (X1,X2)
  print (X)
elif (Setting == 2):
  X1 =float(input("masukkan nilai X1 Interpolasi Linier :"))
  X = Interpolasi_Linier (X1)
  print (X)
elif (Setting == 3):
  X1 =float(input("masukkan nilai X1 Secant :"))
  X = Secant (X1)
  print (X)
elif (Setting == 4):
  X1 =float(input("masukkan nilai X1 Newton Rapson :"))
  X = Newton_Rhapson (X1)
  print (X)
elif (Setting == 5):
  X1 =float(input("masukkan nilai X1 Iterasi :"))
  X = Iterasi (X1)
  print (X)
elif (Setting == 6):
  A = np.array([[5,10,3,8], [1,5,6,6], [2,3,5,4], [3,6,6,5]], dtype=float)
  f = np.array([4.886, 9.086, 20.086, 10.086])
  print('A = \n%s and f = %s' % (A, f))
  B, g = Cal_LU_pivot(A, f)
  print (B,g)
  x = np.linalg.solve(A, f)
  print('Hasil perhitungan gauss adalah x = \n %s' % x)
elif (Setting == 7):
  m = np.array([[5,10,3,8,4.886],
             [1,5,6,6,9.086],
             [2,3,5,4,20.086],
             [3,6,6,5,10.086]],dtype=float)
  n = int(input("masukkan nilai n Gauss jORDAN :"))

  # Menampilkan matrix awal
  print('Matriks Persamaan')
  print(m)

  # Menampilkan Hasil
  m = GaussJordan(m,n)
  print(f"""Hasil Pengolahan menggunakan metode Gauss Jordan didapatkan matriks:
  {m}""")
elif (Setting == 8):
  #Masukan
  A= np.array([[5,10,3,8],
             [1,5,6,6],
             [2,3,5,4],
             [3,6,6,5]],dtype=float)
  b=[4.886, 9.086, 20.086, 10.086]
  x=np.zeros_like(b)
  n=int(input("masukkan nilai n Gauss Seidel :"))
  H= GaussSeidel(A, b, x, n)
  K= solve(A,b)

  print(f'Hasil pengerjaan menggunakan Gauss Seidel didapatkan nilai tiap variabel {H}')
  print(f'Variabel matriks menggunakan SciPy {K}')
elif (Setting == 9):
  A= np.array([[5,10,3,8],
             [1,5,6,6],
             [2,3,5,4],
             [3,6,6,5]])
  b=[4.886, 9.086, 20.086, 10.086]

  x = [1.0,1.0,1.0,1.0]
  n = int(input("masukkan nilai n Jacobi :"))
  J = jacobi(A,b,x,n)
  O = solve(A,b)
  print(f'Hasil pengerjaan menggunakan Jacobi didapatkan nilai tiap variabel {J}')
  print(f'Variabel matriks menggunakan SciPy {O}')
elif (Setting == 10):
  f = lambda x : 3*(x**3) + 4*(x**2) + 6
  a = int(input("masukkan nilai a Jacobi :"))
  b = int(input("masukkan nilai b Jacobi :"))
  N = int(input("masukkan nilai N Jacobi :"))

  x = np.linspace(a,b,N+1)
  y = f(x)

  X = np.linspace(a,b+1,N)
  Y = f(X)
  plt.plot(X,Y)

  for i in range(N):
      xs = [x[i],x[i],x[i+1],x[i+1]]
      ys = [0,f(x[i]),f(x[i+1]),0]
      plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)

  plt.title('Trapesium banyak pias, N = {}'.format(N))
  plt.show()

  L = trapesium(f,a,b,N)
  print(L)
elif (Setting == 11):
  x1 = float(input("batas bawah (x1): "))
  x2 = float(input("batas atas (x2): "))

  hasil = simpson1per3(x1, x2, 2)
  print("nilai integral metode Simpson 1/3:",hasil )
elif (Setting == 12):
  h = float(input("Masukkan nilai h: ")) # Step size
  x0 = float(input("Masukkan nilai x awal: "))
  xn = float(input("Masukkan nilai x akhir: "))
  Nama = input ("Masukkan Nama :")
  X= Euler (h,x0,xn, Nama)
  print (X)
else :
  h = float(input("Masukkan nilai h: ")) # Step size
  x0 = float(input("Masukkan nilai x awal: "))
  xn = float(input("Masukkan nilai x akhir: "))
  Nama = input ("Masukkan Nama :")
  X= Heun (h,x0,xn, Nama)
  print (X)


print ('Disusun Oleh : Kelompok 6 Metnum 2021 \n',
       'Zahra Aninda Pradiva-26050119130090\n',
       'Az Zahrawaani Tegar Fityanto-2600119130089\n',
       'Pramesthi Dwi Octavianna-2600119130086\n',
       'theodorus karel e m-2600119130084\n',
       'M Al faridzie-2600119130083\n',
       'Nauvan Prabhu-2600119130081\n',
       'Reza Aulia Prastiwi-2600119120001\n',
       'Azzaam Muwaffiq-2600119130080\n',
       'Ramadoni Khirtin -2600119130079\n',
       'Kevifa Syama S W-2600119130078\n',
       'Romli Syarifuddin-2600119130077\n',
       'Anisa Lintang Safitri-2600119130076\n')


# In[ ]:




