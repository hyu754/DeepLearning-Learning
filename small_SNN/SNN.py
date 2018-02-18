import numpy as np
from pylab import *
'''
Initialize some variables
'''
dt = 0.5 #ms
a = 0.02
d =8
b =0.2
c = -65

T= 1000

#Voltage trace
v = -70
u = -14

v_array=[]
for t in range(T):
	I_app=0
	if((t<700) and (t>200)):
		I_app=7

	if(v<35):
		v = v +dt*((0.04*v + 5)*v - u +140 +I_app)
		u = u +dt* a*(b*v - u)	
		v_array.append(v)
	elif(v>=35):
		v = 35

		u = u+d
		#update v vector before v=c
		v_array.append(v)
		v=c


figure()
time_vector = [i for i in range(T)]

plot(time_vector,v_array)
show()





	