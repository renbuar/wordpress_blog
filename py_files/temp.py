import time
time_start=time.clock()

found=0
i=20
while found==0:
    i+=20
    found=1
    for j in range (2,20):
        if i%j!=0:
            found=0
            break
print (i)
time_end=time.clock()
print('Time taken (seconds): ',time_end-time_start)