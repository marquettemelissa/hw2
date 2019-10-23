import numpy as np
import matplotlib.pyplot as plt
import sys


def cheb_mat(xlen, order):
	x = np.linspace(-1., 1., xlen)
#	x = np.linspace(0.5, 1., xlen)
	matty = np.zeros((xlen, order+1))
	matty[:,0] = 1.
	if order >= 1:
		matty[:,1] = x
	if order > 1:
		for i in range(1, order):
			matty[:, i+1] = 2*x*matty[:,i]-matty[:,i-1]
	return matty, x

n = 1000
order = 50
matty, x = cheb_mat(n, order)
#y = np.sin(x*np.pi)
#x_reshape = 4.*np.linspace(0.5, 1., n)+3.
#x_reshape = (x/4.)+0.75
y = np.log2(x_reshape)
lefty = np.dot(matty.transpose(), matty)
righty = np.dot(matty.transpose(), y)
params = np.dot(np.linalg.inv(lefty), righty)

matty = matty[:,1::2]
params = params[1::2]

plt.figure()
for cutoff in range(20):

	y_fit = np.dot(matty[:,:cutoff], params[:cutoff]) #(-0.5 to account for weird way I'm casting x)
	resid = y-y_fit
	print np.sum(np.abs(params[:cutoff]))

	plt.plot(x_reshape, y_fit)

plt.plot(x_reshape, y, color='red')
plt.show()

#this isn't working and I am too tired to keep doing it, but hey, there's some code here =)

#sys.exit('no prob 2!')

##########prob 2

#load in the data and chop it up to focus on the flare
flaredat = np.loadtxt('229614158_PDCSAP_SC6.txt',delimiter=',')
#1706.52314133376
t_flare = 1706.52314133376
start = 3200
end = 3300
time = flaredat[start:end, 0]

#make some initial guesses
guess_n0 = 0.25
guess_lamb = 50.
guess_fit = guess_n0*np.exp(-guess_lamb*(time-t_flare)) + 1.

#plot initial guess
plt.figure()
plt.plot(flaredat[start:end,0], flaredat[start:end,1])
plt.plot(time, guess_fit)
plt.xlabel('time')
plt.ylabel('flux')
plt.title('initial guess fit')
#plt.show()



def calc_decay(n0, lamb, t):
	#calculate exponential decay fit
	y = n0*np.exp(-lamb*(t-t_flare)) + 1.
	#calculate gradients needed for chisq
	#grad[:,0] is grad for n0; grad[:,1] is grad for lambda
	grad = np.zeros((t.size,2))
	grad[:,0] = np.exp(-lamb*(t-t_flare))
	#not sure why python is making me squeeze this, but sure, whatever
	grad[:,1] = -n0*(t-t_flare)*np.squeeze(np.asarray(np.exp(-lamb*(t-t_flare))))
	return y, grad

#loop a few times to improve fit
for i in range(5):
	#calculate fit
	y_fit, grad = calc_decay(guess_n0, guess_lamb, time)
	#do linalg stuff
	r = flaredat[start:end, 1] - y_fit
	error = (r**2).sum()
	r = np.matrix(r).transpose()
	grad = np.matrix(grad)
	lefty = grad.transpose()*grad
	righty = grad.transpose()*r
	dparam = np.linalg.inv(lefty)*(righty)
	#update guess parameters -- for some reason python also wants to make me explicitly force these into arrays
	guess_n0 = guess_n0 + np.asarray(dparam[0])
	guess_lamb = guess_lamb + np.asarray(dparam[1])
	print guess_n0, guess_lamb, error

#plot pretty fit
plt.figure()
plt.title('Newton\'s method fit')
plt.plot(flaredat[start:end,0], flaredat[start:end,1])
plt.plot(time, y_fit[0,:])
plt.xlabel('time')
plt.ylabel('flux')
plt.show()
