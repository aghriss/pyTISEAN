import numpy as np

BOX=1024
ibox=BOX-1

def false_nearest(series,minemb=1,maxemb=5,delay=1.0,rt=2.0):
    #series = np.random.normal(0,1,(10000,3))
    inter = 0.0
    eps0 = 1.0e-5
    #series = np.zeros((length,comp))
    comp = series.shape[1] # number of components
    length = series.shape[0]
    maxdim = comp*(maxemb+1) # maximum dimention of the created vector
    mn = series.min(0) # the minimum value per compnents
    ind_inter = series.max(0)-mn # the interval length per component
    series = (series -mn)/ind_inter # rescaling the series per component
    varianz = np.std(series,axis=0).min()**2 # the minimum of variances per components
    inter = ind_inter.max() # the maximum interval length per component
    theiler = 0
    lst = np.zeros(length,dtype=int)
    nearest = np.zeros(length,dtype=int) # is marke =1 when the neasrest neighbor of i if found
    box = np.zeros((BOX,BOX),dtype=int) # 
    
    def mmb(ls,hdim,hemb,eps):
        
        # Groups points in matrix of size 1024x1024
        # If two elments are in distance < esp they should be in neighboring cells
        # The inverse is not necessarily true
        # lst keeps the value of the previous element
        box = -np.ones((BOX,BOX),dtype=int)
        lst = ls.copy()
        for i in range(length-(maxemb+1)*delay):
            x = int(series[i,0]/eps)&ibox
#            print(i+hemb,hdim)
            y = int(series[i+hemb,hdim]/eps)&ibox
            lst[i] = box[x][y]
            box[x][y] = i
        return box,lst
    def find_nearest(n,dim,vcomp,vemb,eps,aveps,vareps,toolarge):
        which = -1
        mindx = 1.1
        
        ic = vcomp[dim]
        ie = vemb[dim]
        x = int(series[n,0]/eps)&ibox
        y = int(series[n+ie,ic]/eps)&ibox
        
        for x1 in range(x-1,x+2):
            x2 = x1&ibox
            for y1 in range(y-1,y+2):
                element = box[x2][y1&ibox]
                while(element != -1):
                    if(np.abs(element-n) > theiler):
                        maxdx = np.abs(series[n,0]-series[element,0])
                        for i in range(1,dim+1):
                            ic=vcomp[i]
                            i1=vemb[i]
                            dx=np.abs(series[n+i1,ic]-series[element+i1,ic])
                            if (dx > maxdx):
                                maxdx=dx
                        
                        if((maxdx < mindx) and (maxdx > 0.0)):
                            which=element
                            mindx=maxdx
                    
                    element=lst[element]
            
            
        if((which != -1) and (mindx <= max(eps,varianz/rt))):
            aveps += mindx
            vareps += mindx*mindx
            factor=0.0
            for i in range(1,comp+1):
                ic=vcomp[dim+i]
                ie=vemb[dim+i]
                hfactor=np.abs(series[n+ie,ic]-series[which+ie,ic])/mindx
                if (hfactor > factor):
                	factor=hfactor
            
            if(factor > rt):
                return 1,aveps,vareps,toolarge+1
            return 1,aveps,vareps,toolarge
        
        return 0,aveps,vareps,toolarge
    # Used to map the dimention with the corresponding delay and compnent
    vcomp = np.zeros(maxdim,dtype=int)
    vemb = np.zeros(maxdim,dtype=int)

    for i in range(maxdim):
        vcomp[i]=int(i%comp)
        vemb[i]=int(int(i/comp)*delay)
    # vcomp = 0,1,3..comp-1, 0,1 2....comp-1 : maxemb+1 times
    # vemb = 0,0,0,0, 1,1,1,1, 2,2,2.... max_emb,max_emb,maxemb.... : comp times each
    
    result = np.zeros((maxemb+1-minemb,4))
        
    for emb in range(minemb,maxemb+1):

        dim = emb*comp-1
        epsilon = eps0
        toolarge = 0
        alldone = 0
        donesofar = 0
        aveps=0.0
        vareps=0.0
        nearest = np.zeros(length,dtype=int)
        while( not alldone and (epsilon < 2.*varianz/rt)):
            alldone = 1
            box,lst=mmb(lst,vcomp[dim],vemb[dim],epsilon)
            for i in range(length-maxemb*delay):
                if not nearest[i]:
                    nearest[i],aveps,vareps,toolarge=find_nearest(i,dim,vcomp,vemb,epsilon,aveps,vareps,toolarge)
                    #print(nearest[i])
                    alldone &= nearest[i]
                    donesofar += nearest[i]

            epsilon*=np.sqrt(2.0)
            if not donesofar:
            	eps0=epsilon
        
        if (donesofar == 0):
            print("Not enough points found for %d!"%emb)
        else:
            print("Done for %d"%emb)
            aveps *= (1./donesofar)
            vareps *= (1./donesofar)
            result[emb]= [dim+1,1.0*toolarge/donesofar,aveps*inter,np.sqrt(vareps)*inter]
    
    return result