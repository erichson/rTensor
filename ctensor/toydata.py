import numpy as np
#import matplotlib as mpl
#mpl.style.available
#mpl.style.use('seaborn-white')    
import matplotlib.pyplot as plt

def toydata(m=100, t=100, background=1, display=1):
    # Define time and space discretizations
    xvals=np.linspace(-50,50,m)
    yvals=xvals 
    xNum = yNum = len(xvals)
    time = np.arange(0,t, 0.7)
    
    
    Xgrid ,Ygrid  = np.meshgrid(yvals,xvals);
    
    def filter2( t, width, tau):
        tFilter = np.ones(t.shape)
        tFilter[t < tau-width/2] = 0
        tFilter[t > tau+width/2] = 0
        return tFilter
    
    # generate background 
    Psi1fn = lambda x, y:  2*np.exp(np.sin(-x**2-y**2))
    a1 = lambda t:  np.ones(t.shape)
    
    sigma = 0.1
    ic2 = [-25, 25]
    Psi2fn = lambda x, y: np.exp(-sigma*(x-ic2[0])**2 - sigma*(y-ic2[1])**2)
    a2 = lambda t:  2*np.cos(t*(2*np.pi)/64);
    
    ic3 = [25, -25]
    Psi3fn = lambda x, y:  np.exp(-sigma/5*(x-ic3[0])**2 - sigma/2*(y-ic3[1])**2)
    a3 = lambda t:  2.5*filter2(t,64,32) * ( np.sin(t*(2*np.pi)/16) )
    
    ic4a = [-10,0]
    ic4b = [10,0]
    Psi4afn = lambda x, y: np.exp(-sigma/2*(x-ic4a[0])**2 - sigma/4*(y-ic4a[1])**2)
    Psi4bfn = lambda x, y:  np.exp(-sigma/2*(x-ic4b[0])**2 - sigma/4*(y-ic4b[1])**2)
    Psi4fn = lambda x, y:  Psi4afn(x,y) + Psi4bfn(x,y)

    ic5 = [-25, -25]
    Psi5fn = lambda x, y:  np.exp(-sigma/5*(x-ic5[0])**2 - sigma/2*(y-ic5[1])**2)
        
    
    a4 = lambda t:  2.3*filter2(t,32,80) * ( np.sin(t*(2*np.pi)/8) )
    
    a5 = lambda t:  1.7*filter2(t,20,60) * ( np.sin(t*(np.pi*4)/8) )
    
    
    Psi1 = Psi1fn(Xgrid,Ygrid)
    Psi2 = Psi2fn(Xgrid,Ygrid)
    Psi3 = Psi3fn(Xgrid,Ygrid)
    Psi4 = Psi4fn(Xgrid,Ygrid)
    Psi5 = Psi5fn(Xgrid,Ygrid)

 
    avec1 = a1(time) 
    avec2 = a2(time) 
    avec3 = a3(time) 
    avec4 = a4(time) 
    avec5 = a5(time) 

 

    # create spatio-temporal tensor
    Tbar = np.zeros((yNum, xNum, len(time)))
    for j in range(len(time)):
        t = time[j]
        frame = avec2[j] * Psi2 + avec3[j] * Psi3 + avec4[j] * Psi4  + avec5[j] * Psi5
        if background==1: frame += avec1[j] * Psi1
        Tbar[:,:,j] = frame


    if display==1: 
        fig1 = plt.figure(num=None,  facecolor='w', edgecolor='k') 
        cmapv = 'gist_ncar'
        plt.subplot(241)
        noisyimg = Psi3.reshape(m,m)
        plt.imshow(noisyimg, cmap=cmapv)
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        
        plt.subplot(242)
        noisyimg = Psi4.reshape(m,m)            
        plt.imshow(noisyimg, cmap=cmapv)
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        
        plt.subplot(245)
        noisyimg = Psi2.reshape(m,m)             
        plt.imshow(noisyimg, cmap=cmapv)
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        
        plt.subplot(246)
        noisyimg = Psi5.reshape(m,m)             
        plt.imshow(noisyimg, cmap=cmapv)
        plt.grid(False)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        
        plt.subplot(243)
        plt.plot(time, avec3, lw=3, color='#d7191c', label='True signal')
        plt.tight_layout()
        #plt.xlabel('$time$', fontsize=24)
        plt.ylabel('', fontsize=12, color='white')
        plt.tick_params(axis='y', labelsize=12) 
        plt.tick_params(axis='x', labelsize=0) 
        plt.xlim([0, t])
        plt.ylim([-4, 4])   
        plt.locator_params(axis='y',nbins=4) 
        plt.locator_params(axis='x',nbins=4)  
        #plt.legend(fontsize=12, loc="lower right")
        plt.grid(False)
        
        plt.subplot(244)
        plt.plot(time, avec4, lw=3, color='#fdae61')
        plt.tight_layout()
        #plt.xlabel('$time$', fontsize=24)
        plt.ylabel('', fontsize=12, color='white')
        plt.tick_params(axis='y', labelsize=12) 
        plt.tick_params(axis='x', labelsize=0) 
        plt.xlim([0, t])
        plt.ylim([-4, 4])   
        plt.locator_params(axis='y',nbins=4) 
        plt.locator_params(axis='x',nbins=4)  
        plt.grid(False)
        
        plt.subplot(247)
        plt.plot(time, avec2, lw=3, color='#abdda4')
        plt.tight_layout()
        #plt.xlabel('$time$', fontsize=24)
        plt.ylabel('', fontsize=12, color='white')
        plt.tick_params(axis='y', labelsize=12) 
        plt.tick_params(axis='x', labelsize=0) 
        plt.xlim([0, t])
        plt.ylim([-4, 4])   
        plt.locator_params(axis='y',nbins=4) 
        plt.locator_params(axis='x',nbins=4)     
        plt.grid(False)
        #plt.xlabel('Time', fontsize=12)

        plt.subplot(248)
        plt.plot(time, avec5, lw=3, color='#2b83ba')
        plt.tight_layout()
        #plt.xlabel('Time', fontsize=12)
        plt.ylabel('', fontsize=12, color='white')
        plt.tick_params(axis='y', labelsize=12) 
        plt.tick_params(axis='x', labelsize=12) 
        plt.xlim([0, t])
        plt.ylim([-4, 4])   
        plt.locator_params(axis='y',nbins=4) 
        plt.locator_params(axis='x',nbins=4)
        plt.grid(False)
        fig1.tight_layout()   
        plt.show()                       



   
    signal = []
    if background==1: signal.append(avec1)
    signal.append(avec2)
    signal.append(avec3)
    signal.append(avec4)
    signal.append(avec5)
    
    
    return (Tbar)