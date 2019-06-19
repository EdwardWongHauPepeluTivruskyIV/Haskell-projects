#Theorie des nombres premiers


import numpy as np
import matplotlib.pylab as plt 

t = np.linspace(2,10,190)
    

#1) Preliminaires

#Renverser une liste

def reverse(t):
    F = len(t)
    L = []
    for k in range(0,F):
        L = L + [t[F-1-k]]
    return(L)
    
#Ordonner une liste par sens decroissant

def tri(t):
    F = len(t)
    L = []
    for i in range(0,F):
        rang = 0
        max = t[0]
        for k in range(0,len(t)):
            if t[k] > max:
                max = t[k]
                rang = k
        L = L + [max]
        del t[rang]
    return(L)
    
#Ordonner une liste par sens croissant

def tri2(t):
    return(reverse(tri(t)))
    
#Nombre de chiffres d'un entier 

def Chiffre(n):
    P = 0
    while n >= 1 :
        n = n/10
        P = P + 1
    return P

#PGCD de a et b
    
def PGCD(a,b):
    S = 0
    for k in range(1,b):
        S = S + int(k*a/b)
    return(a + b - a*b + 2 *S)

def PGCL(a):
    for k in range(a):
        print(k,PGCD(a,k))
        
#Ecriture fractionnaire rÈduite d'un rationnel de pÈriode dÈcimale p
def fract(p):
    return(int(p / PGCD(p, 10**(Chiffre(p))-1)), int((10**(Chiffre(p)) - 1) / PGCD(p, 10**(Chiffre(p))-1)))

#Les des Ècritures fractionnaires
def Jun(n):
    for f in range(1,n):
        print(f,fract(f))
    

def num(x):
    for k in range(2,x):
        for i in ListePremier(k):
            for j in ListePremier(k):
                if k == i**j - j**i:
                    print(k,(i,j))
        
#2) Nombres premiers

#Crit√®re de divisibilit√© par 3 :

def TestDiv3(n):
    V = 0
    for m in range(0, Chiffre(n)):
        V = V + (n // 10**m) % 10
    if V % 3 == 0:
        return 1
    else:
        return 0
        
#Crit√®re de divisibilit√© par 5 :

def TestDiv5(n):
    if (n%10) == 5 or (n%10) == 0:
        return 1
    else :
        return 0

        
#Crit√®re de divisibilit√© par 11 :

def TestDiv11(n):
    M = Chiffre(n)
    S = 0
    for k in range(0,M):
        if k%2 == 0:
            S = S - (n//(10**k))%10
        else:
            S = S + (n//(10**k))%10
    if S/11 == int(S/11):
        return 1
    else:
        return 0

#Test 

def J(n):
    for k in range(1,n):
        print(k, TestDiv3(k))
    
#Ensemble des diviseurs d'un nombre

def ED(n):
    L = []
    if n%2 == 0 and TestDiv3(n) == 0 and TestDiv5(n) == 0 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%3 == 0 or k%5 == 0 or k%11 == 0:
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 0 and TestDiv5(n) == 0 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if  k%3 == 0 or k%5 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 0 and TestDiv5(n) == 1 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%3 == 0 or k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 0 and TestDiv5(n) == 1 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%3 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 1 and TestDiv5(n) == 0 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if  k%5 == 0 or k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 1 and TestDiv5(n) == 0 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%5 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 1 and TestDiv5(n) == 1 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 0 and TestDiv3(n) == 1 and TestDiv5(n) == 1 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            o = n/k
            if o == int(o): 
                L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 0 and TestDiv5(n) == 0 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%3 == 0 or k%5 == 0 or k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 0 and TestDiv5(n) == 0 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%3 == 0 or k%5 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 0 and TestDiv5(n) == 1 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%3 == 0 or k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 0 and TestDiv5(n) == 1 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%3 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 1 and TestDiv5(n) == 0 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%5 == 0 or k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 1 and TestDiv5(n) == 0 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%5 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 1 and TestDiv5(n) == 1 and TestDiv11(n) == 0:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 or k%11 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    if n%2 == 1 and TestDiv3(n) == 1 and TestDiv5(n) == 1 and TestDiv11(n) == 1:
        for k in range(1,int((n+1)**0.5)+1) :
            if k%2 == 0 :
                L = L
            else :
                o = n/k
                if o == int(o):
                    L = L + [k] + [int(o)]
    return(tri2(L))
    
#Liste des nombres premiers infÈrieurs ‡ n

def ListePremier(n):
    L = []
    for k in range(2,n+1):
        if ED(k) == [1, k]:
            L = L + [k]
    return(L)

#Liste des n premiers nombres premiers 
    
def Premier2(n):
    L = []
    k = 2
    while len(L) != n:
        if ED(k) == [1, k]:
            L = L + [k]
        k += 1
    return(L)

#Test nombre premier

def Testpremier(n):
    if ED(n) != [1,n] or n == 1:
        return 0
    else:
        return 1

def diz(n):
    for k in range(1,n):
        if Testpremier(10**k+1) == 1:
            print(10**k+1, "premier")
            
#Proportion de nombres premiers

def Prop0(n):
    P = 0
    S = 0
    L = []
    for k in range(1,n+1):
        if Testpremier(k) == 1:
            P = P + 1
        S = S + 1
        L = L + [P/S]
    return(L)
    
#Nombre de nombres premier

def Nb(n):
    S = 0
    L = []
    for k in range(1,n+1):
        if Testpremier(k) == 1:
            S = S + 1
        L = L + [S]
    return(L)


def K(n):
    L = []
    S = 0
    for k in range(1,n+1):
        L = L + [(k * (1/(np.log(k)))) / S]
        S = S + 1
    return(L)
    
def Q(n):
    L = []
    for k in range(1,n+1):
        L = L + [(k * (1/(np.log(k))))]
    return(L)

#sinus cardinal

def sinc(x):
    if x == 0 :
        return(1)
    else :
        return(np.sin(x)/x)

def M(P):
    L = []
    for k in P:
        L = L + [sinc(k)]
    return(L)
    
#plt.plot(t, M(t), 'r')

#3 Decomposition en facteurs premiers

#D√©composition en facteurs premiers d'un nombre 

def DecompPremier(n):
    C = []
    for k in ListePremier(n):
        while int(n/k) == n/k:
            n = n/k
            C = C + [k]
    return(C)

#Liste des nombres avec leur decomp en facteurs premiers
    
def ListDecop(n):
    for k in range(n+1):
        print(k,DecompPremier(k))
        
#Nombre d'entier dans la decomp en facteurs premiers
        
def Composabilite(n):
    for k in range(10000,n+1):
        print(k, len(DecompPremier(k)))

# 4- Autre
def sigma(n):
    s = 1
    for k in range(1,n):
        s = s + 10**k
        print(s - (10**(n/2)+1)**(2))

#5 Nombres hexaÈdriques
        
#Plus petit entier tel que n-k et n+k soit premier

def r0(n):
    k = 0
    while Testpremier(n-k) == 0 or Testpremier(n+k) == 0:
        k = k+1
    return(k)

def Lister0(m):
    for i in range(1,m):
        if Testpremier(i) == 0 :
            print((i,r0(i)))

def hexa(p):
    if 6%r0(p) == 0:
        return(1)
    else:
        return(0)

def nbhexa(x):
    k = 2
    res = 0
    while k <= x:
        if Testpremier(k) == 0 and hexa(k) == 1:
            res = res + 1
        k = k + 1
    return(res)

def ke(n):
    for k in range(4,n+1):
        if Testpremier(k) == 0 and hexa(k) == 1:
            print((k,r0(k)))

def r01(x):
    k = 2
    res = 0
    while k < x:
        if Testpremier(k) == 0 and r0(k) == 1 and hexa(k) == 1:
            res = res + 1
        k = k + 1
    return(res)
    
    
def proportionr0(x):
    return(r01(x)/nbhexa(x))

def Lhd(x):
    for k in range(5,x):
        print((k,proportionr0(k)))
    
def Lis(x):
    L = []
    for k in range(4,x):
        L = L + [proportionr0(k)]
    return(L)

def Jk(n):
    L = []
    for k in range(5,n+1):
        L = L + [(proportionr0(k)-1/6)*np.log(k)]
    return(L)

#6- Polynomes premiers
    
#Polynome qui a ses n premiers racines des nombres premiers
    
def polynome(n):
    return(np.poly(Premier2(n)))

#Evaluation en x

def Pol(n,x):
    return(np.polyval(polynome(n),x))
    
def Polylist(n):
    L = []
    for k in range(len(t)):
        L = L + [Pol(n,k)]
    return(L)

#for v in range(1,10):    
#plt.plot(t,Polylist(v),'r')
#plt.show()
#plt.plot(t, K(len(t)), 'b') 
#plt.plot(t,Lis(1004),'r')
#plt.show()
    
#7 - Parties premiËre et dÈcomposition premiËre

#partie premiere par exces
    
def Pplus(x):
    L = ListePremier(x+200)
    n = 2
    k = 0
    while n < x:
        k += 1
        n = L[k]
    return(n)

#partie premiere par defaut
    
def Pmoins(x):
    if x <= 1:
        return(0)
    else:
        L = ListePremier(x+200)
        n = 2
        k = 0
        while n < x:
            k += 1
            n = L[k]
        if n == x:
            return(n)
        else:
            return(L[k-1])
        
#decomposition premiere
        
def Decop(n):
    L = [Pmoins(n)]
    p = n - Pmoins(n)
    while p != 0:
        if p != 1:
            L = L + [Pmoins(p)]
            p = p - Pmoins(p)
        else:
            L = L + [1]
            p = 0
    return(L)

#Liste des decompositions premiËres
    
def Kim(n):
    for k in range(2,n):
        print(k,Decop(k))
        
#Liste des decompositions meilleures version

def Kim2(n):
    M = Premier2(2*n)
    for k in range(1,n):
        p = 1
        d = 0
        r = 0
        while p <= k:
            r = p
            p = M[d]
            d += 1
        L = [r]
        g = k - r
        while g != 0:
             a = 1
             b = 0
             c = 0
             while a <= g:
                 c = a
                 a = M[b]
                 b += 1
             L = L + [c]
             g = g - c
        print(k,L)

def Compt(x):
    p = 0
    for k in range(2,x+1):
        if Pmoins(x+k) <= Pmoins(x) + Pmoins(k):
            p += 1
    return(p)
    
def acs(x):
    print(x, (Compt(x)/(x-1),(1 - Compt(x)/(x-1))))

def Propliste(x):
    return((Compt(x)/(x-1)))
    
def Prop(x):
    for k in range(101,x+1):
        print(k, (Compt(k)/(k-1),(1 - Compt(k)/(k-1))))
    
def Compt3(x):
    L = ListePremier(Pplus(2*x))
    g = Pmoins(x)
    p = 2
    q = 2
    res = 0
    for k in range(2,x+1):
        i = 0
        while p < k:
            i += 1
            p = L[i]
        if p != k:
            p = L[i-1]
        j = 0
        while q < x+k:
            j += 1
            q = L[j]
        if q != x+k:
            q = L[j-1]
        if q < p + g:
            res += 1
    return(p)

def Prop2(x):
    for k in range(2,x+1):
        print(k, (Compt3(k)/(k-1),(1 - Compt3(k)/(k-1))))

def Hprem(x):
    L = []
    for k in range(2,x+2):
        L = L + [Propliste(k)]
    return(L)

#Description des entiers pairs extrÈmaux jusqu'‡ n

def Descripp(p,n):
    for k in range(p,n+1):
        if Testpremier(k) == 1:
            print(k,"premier")
        if k/2 == int(k/2) and Testpremier(k - Pmoins(k)) == 1:
            print(k,1)
        if k/2 == int(k/2) and Testpremier(k - Pmoins(k)) != 1:
            print(k,0)
            
            
def Descripp2(n):
    for k in range(3,n+1):
        if k/2 == int(k/2) and (k - Pmoins(k) != 1):
            print(k, Testpremier(k - Pmoins(k)))

#plt.plot(t, Hprem(len(t)), 'b') 

#Psi(n) : Ècart de n au milieu de se pÈriphÈrie premiËre
            
def psi(n):
    return(int(n - (Pplus(n) + Pmoins(n))/2))
    
def psiliste(n):
    for k in range(2,n):
        print(k, psi(k))

#Partie premiËre des nombres harmonique
        
def Harmonique(n):
    res = 0
    for k in range(1,n):
        res = res + 1/k
    return(res)
    
def PPHarm(n):
    return(Pmoins(int(Harmonique(n))))

def lharm(n):
    for k in range(3,n):
        print(k, PPHarm(k))

def primec(n,m):
    L = []
    for k in ListePremier(n):
        L += [(-1)**(k+1) * k]
    return(L)

def primecb(n):
    L = []
    for k in ListePremier(n):
        L += [3**k]
    return(L)
    
def impair(n):
    L = []
    for k in range(n+1):
        L += [2**(2*k+1)]
    return(L)
    
def sinv(L):
    res = 0
    for k in L:
        res += 1/k
    return(res)

def suit(n):
    for k in range(1,n+1):
        print(sinv(primec(1000,k)))
        
        
        
#Conjecture de Goldbach

#nombre de maniËre d'Ècrire n comme somme de deux nombres premiers
        
def Prime(n):
    res = 0
    L = ListePremier(n)
    for k in L:
        for p in L:
            if p <= k and k+p == n:
                res = res + 1
    return(res)

def Goldbach(n):
    L = []
    for k in range(2,n+1):
        L += [Prime(k)]
    return(L)

#plt.plot(t, Goldbach(len(t)), 'b') 
#plt.show()
    
#Suite avec rayon de primalitÈ

def suite(n,a):
    u = a
    for k in range(n+1):
        u = u * r0(u)
        print(u)
    
def ope(x,y):
    return(int((y-x)**(x)/y**(1/x)))
            
            
def lope(n):
    L = ListePremier(n)
    for k in range(1,len(L)-1):
        print((L[k-1],L[k+1]),ope(L[k-1],L[k+1]))
    
    
def sophie(n):
    for k in range(2,n):
        print(k,Testpremier(1+(2*k**2)**2))

#primorielle
        
def primorielle(n):
    L = Premier2(n+1)
    res = 1
    for k in range(1,n+1):
        res = L[k] * res
    return(res)

def baba(x):
    return(1 + 1/(np.log(primorielle(x))-1)**x)
    
def listbaba(n):
    L = []
    for k in range(n):
        L += [baba(k)]
    return(L)

#plt.plot(t, listbaba(len(t)), 'b') 
#plt.show()
    
#Polynomes de Lagrange de nombres premiers

def product(L,x,j):
    prod = 1
    for k in range(len(L)):
        if k != j:
            prod = prod * (x-L[k][0])/(L[j][0] - L[k][0])
    return(prod)
        

def Lagrange(L,x):
    res = 0
    for j in range(len(L)):
        res = L[j][1] * product(L,x,j)
    return(res)

def Cor(n):
    L = []
    M = Premier2(n)
    for k in range(1,n):
        L += [[k,M[k-1]]]
    return(L)

def LLag(n,m):
    L = []
    H = Cor(n)
    for k in range(m):
        L += [Lagrange(H,k)]
    return(L)

#for k in range(1,25):
    #plt.plot(t, LLag(k,len(t)), 'b') 
#plt.show()

def f1(x,n):
    return(np.log(np.log(Lagrange(Cor(n),x)/x))/(np.log(x)))
    
    
    
    
    
    
    