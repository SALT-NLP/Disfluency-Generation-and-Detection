import sys
import numpy as np
from scipy import stats

def computeF(pc1, pc2, rc1, rc2):
    p=sum(pc1)/sum(pc2)
    r=sum(rc1)/sum(rc2)
    f=2*p*r/(p+r)
    return p,r,f

#Bootstrap
#Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
# let r be the number of times that delta(A,B)<2*orig_delta(A,B). significance level: r/R
# This implementation follows the description in Berg-Kirkpatrick et al. (2012), 
# "An Empirical Investigation of Statistical Significance in NLP".
def Bootstrap(data_A_pc1, data_A_pc2, data_A_rc1, data_A_rc2, data_B_pc1, data_B_pc2, data_B_rc1, data_B_rc2, n, R):
    pA,rA,fA=computeF(data_A_pc1, data_A_pc2, data_A_rc1, data_A_rc2)
    pB,rB,fB=computeF(data_B_pc1, data_B_pc2, data_B_rc1, data_B_rc2)
    print('p: ',pA,pB)
    print('r: ',rA,rB)
    print('f: ',fA,fB)
    
    delta_orig_r =rA-rB
    delta_orig_f =fA-fB
    rr = 0
    rf = 0
    for x in range(0, R):
        if x%5000==0:
            print(x)
        samples = np.random.randint(0,n,n) #which samples to add to the subsample with repetitions
        ap1=[]
        ap2=[]
        ac1=[]
        ac2=[]
        bp1=[]
        bp2=[]
        bc1=[]
        bc2=[]
        for samp in samples:
            ap1.append(data_A_pc1[samp])
            ap2.append(data_A_pc2[samp])
            ac1.append(data_A_rc1[samp])
            ac2.append(data_A_rc2[samp])
            bp1.append(data_B_pc1[samp])
            bp2.append(data_B_pc2[samp])
            bc1.append(data_B_rc1[samp])
            bc2.append(data_B_rc2[samp])
        pA,rA,fA=computeF(ap1, ap2, ac1, ac2)
        pB,rB,fB=computeF(bp1, bp2, bc1, bc2)
        delta_r = rA-rB
        delta_f = fA-fB
        if (delta_r > 2*delta_orig_r):
            rr = rr + 1
        if (delta_f > 2*delta_orig_f):
            rf = rf + 1
    pval_r = float(rr)/(R)
    pval_f = float(rf)/(R)
    return pval_r, pval_f




def main():
    if len(sys.argv) < 3:
        print("You did not give enough arguments\n ")
        sys.exit(1)
    filename_A = sys.argv[1]
    filename_B = sys.argv[2]
    alpha = sys.argv[3]

    data_A_pc1=[]
    data_A_pc2=[]
    data_A_rc1=[]
    data_A_rc2=[]
    data_B_pc1=[]
    data_B_pc2=[]
    data_B_rc1=[]
    data_B_rc2=[]
    with open(filename_A) as f:
        for line in f:
            items=line.strip().split()
            items=list(map(float,items))
            data_A_pc1.append(items[0])
            data_A_pc2.append(items[1])
            data_A_rc1.append(items[2])
            data_A_rc2.append(items[3])

    with open(filename_B) as f:
        for line in f:
            items=line.strip().split()
            items=list(map(float,items))
            data_B_pc1.append(items[0])
            data_B_pc2.append(items[1])
            data_B_rc1.append(items[2])
            data_B_rc2.append(items[3])

    print("\nBootstrap")
    name = "Bootstrap"#input("\nEnter name of statistical test: ")

    if(name=="Bootstrap"):
        R = max(10000, int(len(data_A_pc1) * (1 / float(alpha))))
        pval_r, pval_f = Bootstrap(data_A_pc1, data_A_pc2, data_A_rc1, data_A_rc2, data_B_pc1, data_B_pc2, data_B_rc1, data_B_rc2, len(data_A_pc1), R)
        if (float(pval_r) <= float(alpha)):
            print("\nTest r result is significant with p-value: {}".format(pval_r))
        else:
            print("\nTest r result is not significant with p-value: {}".format(pval_r))
        if (float(pval_f) <= float(alpha)):
            print("\nTest f result is significant with p-value: {}".format(pval_f))
        else:
            print("\nTest f result is not significant with p-value: {}".format(pval_f))
        return

    else:
        print("\nInvalid name of statistical test")
        sys.exit(1)





if __name__ == "__main__":
    main()










