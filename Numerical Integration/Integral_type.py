import numpy as np
import scipy.integrate
from scipy.integrate import cumulative_trapezoid, quad, simpson as scipy_simpson
import matplotlib.pyplot as plt

def trapezoidal(fx,a,b,n):
    if (n<0):
        n= -n
    if (n%1!=0):
        n/=1
    h= (b-a)/n
    suma = fx(a)/2 + fx(b)/2
    for i in range (1,n):
        suma+=fx(a+i*h)
    return suma * h

def simpson(fx,a,b,n):
    if (n<0):
        n= -n
    if (n%1!=0):
        n/=1
    if (n%2!=0):
        n+=1
    h=(b-a)/n
    suma= fx(a) + fx(b)
    for i in range(1,n):
        if(i%2==0):
            suma+=2*fx(a+i*h)
        else:
            suma+=4*fx(a+i*h)
    return suma *h /3

def simpson_zad2(x,y):
    n = len(x)-1
    a=x[0]
    odd = False
    if (n%2!=0):
        odd=True
        n-=1
    b=x[n]
    h= (b-a) / n
    suma= y[0]+y[n]
    for i in range(1,n):
        if (i % 2 == 0):
            suma += 2 * y[i]
        else:
            suma += 4 * y[i]
    suma = suma*h/3
    if odd:
        suma = suma + (x[n+1]-x[n])*(y[n]+y[n+1])/2
    return suma


def f1(x):
    return np.exp(-x**2) * (np.log(x))**2

def f2(x):
    return 1 / (x**3 - 2*x - 5)

def f3(x):
    return (x**5) * np.exp(-x) * np.sin(x)

def f4(y, x):
    suma_xy = x + y
    suma_xy = np.where(suma_xy <= 0, 1e-15, suma_xy)
    return 1 / (np.sqrt(suma_xy) * (1 + suma_xy))

def f5(x, y):
    return x**2 + y**2

def porownanie():
    przypadki = [
        {"funkcja": f1, "a": 0, "b": np.pi, "nazwa": "sin(x)"},
        {"funkcja": f2, "a": 0, "b": 2, "nazwa": "exp(-x)"},
        {"funkcja": f3, "a": 1, "b": 4, "nazwa": "x^3 - 2x^2 + 5"}
    ]
    for punkty in [11, 12]:  # 11 punktów = 10 podprzedziałów, 12 punktów = 11 podprzedziałów
        print(f"\n=======================================================")
        print(f"--- Test dla liczby węzłów: {punkty} (liczba podprzedziałów: {punkty - 1}) ---")
        print(f"=======================================================")

        for p in przypadki:
            fx = p["funkcja"]
            a, b = p["a"], p["b"]

            # Generowanie węzłów za pomocą np.linspace
            x_wenty = np.linspace(a, b, punkty)
            y_wenty = fx(x_wenty)

            # 1. Twój Simpson
            wynik_moj = simpson_zad2(x_wenty, y_wenty)

            # 2. SciPy Simpson
            wynik_scipy_simps = scipy_simpson(y_wenty, x=x_wenty)

            # 3. SciPy Trapezoid
            wynik_scipy_trapezoid = scipy.integrate.trapezoid(y_wenty, x=x_wenty)

            # 4. NumPy Trapezoid (od NumPy 2.0 zastępuje dawne np.trapz)
            wynik_numpy_trapezoid = np.trapezoid(y_wenty, x=x_wenty)

            # 5. SciPy Cumulative Trapezoid (całkowanie skumulowane)
            # Zwraca tablicę całek dla kolejnych kroków. Ostatni element to całka na całym przedziale [a, b].
            wynik_cum_trapezoid_array = cumulative_trapezoid(y_wenty, x=x_wenty, initial=0)
            wynik_scipy_cum_trap = wynik_cum_trapezoid_array[-1]

            # 6. SciPy Quadrature (kwadratura Gaussa)
            # Przyjmuje funkcję (a nie tablicę y) oraz granice. rtol podkręcamy pod liczbę punktów.

            # 7. Referencja: SciPy Quad
            wynik_quad, _ = quad(fx, a, b)

            # Obliczanie błędów bezwzględnych w porównaniu do quad
            print(f"\nFunkcja: {p['nazwa']} na [{a:.2f}, {b:.2f}]")
            print(f"  -> Twój Simpson:          {wynik_moj:.8f} (Błąd: {abs(wynik_moj - wynik_quad):.2e})")
            print(
                f"  -> SciPy Simpson:         {wynik_scipy_simps:.8f} (Błąd: {abs(wynik_scipy_simps - wynik_quad):.2e})")
            print(
                f"  -> SciPy Trapezoid:       {wynik_scipy_trapezoid:.8f} (Błąd: {abs(wynik_scipy_trapezoid - wynik_quad):.2e})")
            print(
                f"  -> NumPy Trapezoid:       {wynik_numpy_trapezoid:.8f} (Błąd: {abs(wynik_numpy_trapezoid - wynik_quad):.2e})")
            print(
                f"  -> SciPy CumTrapezoid:    {wynik_scipy_cum_trap:.8f} (Błąd: {abs(wynik_scipy_cum_trap - wynik_quad):.2e})")
            print(f"  -> SciPy Quad [REF]:      {wynik_quad:.8f}")
            print("-" * 60)

def double_integral():
    print()


def test_double():
    wynik, blad = scipy.integrate.dblquad(f4,0,1,0,lambda x: 1-x)
    print("Scipy.dblquad podaje wartość: ", wynik, " i szacuje błąd: ", blad," a realna wartość to ", 2-np.pi/2)


# def gauss_legendre(fx,a,b,n):
#     suma =0
porownanie()
print("\n")
test_double()
