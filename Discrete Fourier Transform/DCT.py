import numpy as np

def rek(x):

    n = len(x)
    if n<=1:
        return x

    F = rek(x[0::2])
    G = rek(x[1::2])

    exp = np.exp(-2j * np.pi * np.arange(n/2) / n )
    wynik = F + G * exp
    wynik= np.hstack(( wynik , F - G * exp ))

    return wynik

sygnal = [4.0, 7.0, 11.0, 10.0, 52.0, 0.0, 71.0, 0.0]
print("Nasze DCT: ", np.round(rek(sygnal), 2))
print("Wbudowane: ", np.round(np.fft.fft(sygnal), 2))