import numpy as np
import os
from random import randint
import random
import matplotlib.pyplot as plt
import bsline
import itertools
import pathlib


def grid():
	global resolution,half_length
	
	gap=0.6
	tick=np.arange(-half_length,half_length+5e-2,gap)
	
	boundary,inner={},{}
	b_key,in_key=0,0
	
	
	for itemx in tick:
		for itemy in tick:
			if (abs(itemx)-half_length)*(abs(itemy)-half_length)<1e-6:
				boundary[b_key]=[itemx,itemy]
				b_key+=1
			else:
				inner[in_key]=[itemx,itemy]
				in_key+=1
				
	total={}
	for keys,values in boundary.items():
		total[keys]=values
	for keys,values in inner.items():
		total[keys+b_key]=values

	neighborlist={}
	for keys,values	in total.items():
		list=[]
		for kk,vv in total.items():
			if (values[0]-vv[0])**2.+(values[1]-vv[1])**2.<(gap*1.5)**2.:
				list.append(kk)
		neighborlist[keys]=set(tuple(list))
			
	return total,neighborlist,boundary
	
def single_line(total,neighborlist,begin_point,boundary_key):
	line=[begin_point]
	next_point=random.sample(neighborlist[begin_point],1)[0]
	line.append(next_point)
	present_point=next_point
	
	while next_point>=boundary_key:
		next_point=random.sample(neighborlist[present_point],1)[0]
		time=0
		while next_point in line:
			next_point=random.sample(neighborlist[present_point],1)[0]
			time+=1
			if time>10:
				break
		line.append(next_point)
		
		present_point=next_point	
	lines=set([])
	flag=False
	if len(line)>40 and len(line)<60:
		lines.add(tuple(line))
		flag=True
	
	if flag:
		cv=np.zeros((len(line),2))
		m=0
		for point in line:
			cv[m][0]=total[point][0]
			cv[m][1]=total[point][1]
			m+=1
		
		p = bsline.bspline(cv,n=int(len(line)*0.8),degree=3)
		while len(p)>5:
			p = bsline.bspline(p,n=int(len(p)*0.8),degree=3)
		
		p = bsline.bspline(p,n=min(int(len(p)*40),200),degree=8)

		xx_d,yy_d=p.T
		return xx_d,yy_d
	else:
		return False
	
def cal_den_line_count(xx_d,yy_d):
	global resolution,half_length
	gap=half_length*2/(resolution)
	center_reso=int(resolution/2)
	density=np.zeros((resolution,resolution,2))
	
	count=0
	grid=np.linspace(-half_length,half_length,resolution+1)
	
	inter_point=[[xx_d[0],yy_d[0],0.]]
	for i in range(1,len(xx_d)):
		mid_point=[(xx_d[i]+xx_d[i-1])*0.5,(yy_d[i]+yy_d[i-1])*0.5]
		for itemx in grid:
			if (xx_d[i]-itemx)*(xx_d[i-1]-itemx)<=0 :
				inter_point.append(mid_point)
		for itemy in grid:
			if (yy_d[i]-itemy)*(yy_d[i-1]-itemy)<=0:
				inter_point.append(mid_point)
	inter_point.append([xx_d[-1],yy_d[-1]])			
	
	for i in range(1,len(inter_point)):	
		direction=np.zeros(2)
		direction[0]=(inter_point[i][0]-inter_point[i-1][0])
		direction[1]=(inter_point[i][1]-inter_point[i-1][1])
		
		pos_x=(inter_point[i][0]+inter_point[i-1][0])/2.
		pos_x=int((pos_x+(half_length-1e-2))/gap)
		pos_y=(inter_point[i][1]+inter_point[i-1][1])/2.
		pos_y=int((pos_y+(half_length-1e-2))/gap)

		density[pos_x,pos_y,0]+=direction[0]
		density[pos_x,pos_y,1]+=direction[1]

	absdensity=np.linalg.norm(density,axis=2)
	
	count_map=np.ones((resolution,resolution))*(absdensity>1e-6)
	return density,count_map,absdensity
	
def multi_lines(n_dis_low,total,neighborlist,boundary):
	global resolution
	n_dis=np.random.randint(n_dis_low*5,max(n_dis_low*20,20))
	boundary_key=len(boundary)
	begin_points=np.random.randint(0,boundary_key,n_dis)
	lines=set([])
	line,plot_coor=[],[]
	density=np.zeros((resolution,resolution,2))
	count_map=np.zeros((resolution,resolution))
	rhototal=np.zeros((resolution,resolution))
	i=0
	while i<n_dis:
		tmp=single_line(total,neighborlist,begin_points[i],boundary_key)
		while not tmp:
			tmp=single_line(total,neighborlist,begin_points[i],boundary_key)
		plot_coor.append(tmp)
		densityplus,count_mapplus,rhototalplus=cal_den_line_count(tmp[0],tmp[1])
		density+=densityplus
		count_map+=count_mapplus
		rhototal+=rhototalplus
		i+=1
	absdensity=np.linalg.norm(density,axis=2)
	return density,count_map,absdensity,plot_coor
	
def plot_line(plot_coor,total,density,count_map,rhototal):		
	global resolution,half_length
	a=1e-5 #10 um
	burgers=3e-10
	area=(half_length*2.*a/resolution)**2.
	rhototal=rhototal*a/(area*burgers)
	print('avage rhototal = %.4e'%(np.mean(rhototal)))
	absdensity=np.linalg.norm(density,axis=2)
	f, (ax1, ax2, ax3) = plt.subplots(1, 3,sharey=True,figsize=(18, 5),dpi=500,tight_layout=True)
	for i in range(len(plot_coor)):
		ax1.plot(plot_coor[i][0], plot_coor[i][1],'-')

	x_d=np.linspace(-half_length+half_length/resolution, half_length-half_length/resolution, resolution)
	y_d=np.linspace(-half_length+half_length/resolution, half_length-half_length/resolution, resolution)

	for i in range(len(x_d)):
		for j in range(len(y_d)):
			if absdensity[i,j]>1e-8:
				ax1.arrow( x_d[i]-0.2*density[i,j,0], y_d[j]-0.2*density[i,j,1], 0.4*density[i,j,0], 0.4*density[i,j,1],fc="k", ec="k",head_width=0.05, head_length=0.08)
				ax2.arrow( x_d[i]-0.2*density[i,j,0], y_d[j]-0.2*density[i,j,1], 0.4*density[i,j,0], 0.4*density[i,j,1],fc="k", ec="k",head_width=0.05, head_length=0.08)
	
	X,Y= np.meshgrid(x_d, y_d)
	print(np.amax(count_map))
	count_map=count_map.T

	from matplotlib.ticker import LogFormatter
	import matplotlib.colors as colors
	data=np.flip(rhototal.T,0)
	min_data=data.copy()
	min_data[min_data<1]=1e18

	print(np.amin(data),np.amin(data)>0)

	cs=ax3.imshow(data,extent=[-half_length,half_length,-half_length,half_length],aspect='auto',
		cmap=plt.cm.jet,interpolation='none',alpha=0.9,
		norm=colors.LogNorm(vmin=min_data.min(), vmax=data.max()))
	formatter=LogFormatter(10,labelOnlyBase=True)

	thresh = count_map.max() *0.25

	for i, j in itertools.product(range(count_map.shape[0]), range(count_map.shape[1])):
		if int(count_map[i, j])>0.75:
			ax3.text(y_d[j], x_d[i], int(count_map[i, j]),
				 horizontalalignment="center",verticalalignment='center',
				 color="black" )#if count_map[i, j] < thresh or count_map[i, j] > 3*thresh else "black")
	f.colorbar(cs, ax=ax3,#ticks=[0.1e14,0.5e14,1e14,5e14,10e14,20e14,50e14,100e14],#, ticks=[item for item in range(0,int(np.amax(count_map))+1)])
				format=formatter)

	ax1.set_xticks(np.linspace(-4.5, 4.5, resolution+1))
	ax1.set_yticks(np.linspace(-4.5, 4.5, resolution+1))
	ax1.set_xlim([-4.5,4.5])
	ax1.set_ylim([-4.5,4.5])
	ax1.grid(color='gray', linestyle='-', linewidth=0.5)
	ax1.set_xticklabels([])
	ax1.set_yticklabels([])

	ax2.set_xticks(np.linspace(-4.5, 4.5, resolution+1))
	ax2.set_yticks(np.linspace(-4.5, 4.5, resolution+1))
	ax2.set_xlim([-4.5,4.5])
	ax2.set_ylim([-4.5,4.5])
	ax2.grid(color='gray', linestyle='-', linewidth=0.5)
	ax2.set_xticklabels([])
	ax2.set_yticklabels([])

	ax3.set_xticks(np.linspace(-4.5, 4.5, resolution+1))
	ax3.set_yticks(np.linspace(-4.5, 4.5, resolution+1))
	ax3.set_xlim([-4.5,4.5])
	ax3.set_ylim([-4.5,4.5])
	ax3.grid(color='gray', linestyle='-', linewidth=0.5)
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	f.savefig('plot.jpeg')#plt.show()
	return f

def crop(density,count_map,rhototal_map,max_label):
	density_temp=np.zeros((len(density),len(density),3))
	density_temp[:,:,:2]=density
	density_temp[:,:,2]=(density[:,:,0]**2.0+density[:,:,1]**2.0)**0.5
	density_temp=density_temp.reshape((-1,len(density),len(density),3))

	s_density_dic={}
	s_rhototal_dic={}
	for i in range(3,len(density)-3):
		for j in range(3,len(density)-3):
			key=int(count_map[i,j])
			center_rhototal=rhototal_map[i,j]
			if key>0 and key<=max_label:				
				if key not in s_density_dic:
					s_density_dic[key]=density_temp[:,i-3:i+4,j-3:j+4,:]
					s_rhototal_dic[key]=center_rhototal*np.ones(1)
				else:
					s_density_dic[key]=np.concatenate((s_density_dic[key],density_temp[:,i-3:i+4,j-3:j+4,:]),axis=0)
					s_rhototal_dic[key]=np.concatenate((s_rhototal_dic[key],center_rhototal*np.ones(1)),axis=0)

	length_vec=np.zeros(max_label)
	for key,value in s_density_dic.items():
		length_vec[key-1]=len(value)
	low=max(np.amin(length_vec)*0.8,1)
	high=max(np.amin(length_vec)+1,1)
	r_density_dic={}
	r_rhototal_dic={}
	if high>low:
		for key,value in s_density_dic.items():
			np.random.shuffle(value)
			
			keep_length=np.random.randint(low,high)
			
			r_density_dic[key]=value[:keep_length]
			r_rhototal_dic[key]=s_rhototal_dic[key][:keep_length]
		
	return r_density_dic,r_rhototal_dic
	
def generate_sample(n_dis,sample_size,max_label,type):
	global resolution
	density_dic={}
	rhototal_dic={}
	max_iter=int(1e7)
	total,neighborlist,boundary=grid()
	for iter in range(max_iter):
		density,count_map,rhototal_map,_=multi_lines(n_dis,total,neighborlist,boundary)
		s_density_dic,s_rhototal_dic=crop(density,count_map,rhototal_map,max_label)
		
		dic_length=np.zeros(max_label)
		for key,value in s_density_dic.items():
			if key not in density_dic:
				density_dic[key]=value
				rhototal_dic[key]=s_rhototal_dic[key]
				# print(value.shape)
			elif len(density_dic[key])<=sample_size:
				density_dic[key]=np.concatenate((density_dic[key],value),axis=0)
				rhototal_dic[key]=np.concatenate((rhototal_dic[key],s_rhototal_dic[key]),axis=0)
			
		for key,value in density_dic.items():
			dic_length[key-1]=len(value)
		if np.amin(dic_length)>sample_size:
			print(iter)
			break
		if iter%1000==0:
			print(iter, np.amin(dic_length),np.amax(dic_length))
			
	for key,value in density_dic.items():
		init_file_name=os.getcwd()+"\\input\\total_"+type+"_"+str(key)+".npz"
		pathlib.Path(os.getcwd()+"\\input").mkdir(parents=True, exist_ok=True)
		print(init_file_name)
		density_save=value[:sample_size]
		rhototal_save=rhototal_dic[key][:sample_size]
		count_save=np.ones(len(density_save))*key
		print(len(density_save))
		print(density_save.shape,rhototal_save.shape,count_save.shape)

		np.savez(init_file_name, density_save, count_save, rhototal_save)

def conta_single(path,n,type):
	for i in range(1,1+n,1):
		init_file_name=path+"\\total_"+type+"_"+str(i)+".npz"
		init_file=np.load(init_file_name)

		if i==1:
			density=init_file['arr_0']
			count=init_file['arr_1'].astype(int)
			rhototal=init_file['arr_2']
		else:
			density=np.append(density, init_file['arr_0'],axis=0)
			count=np.append(count, init_file['arr_1'],axis=0)
			rhototal=np.append(rhototal, init_file['arr_2'],axis=0)
		init_file.close()
	return density,count,rhototal
		
def conta_file(path,n):
	density,count,rhototal=conta_single(path,n,'train')
	t_density,t_count,t_rhototal=conta_single(path,n,'validation')
	# test_density,test_count,test_rhototal=conta_single(path,n,'test')
	init_file_name=path+"\\crop_1_"+str(n)+"_total_comb.npz"
	np.savez(init_file_name, density=density, count=count,rhototal=rhototal,
			t_density=t_density,t_count=t_count,t_rhototal=t_rhototal)

def demo_plot(n_dis):
	total,neighborlist,boundary=grid()
	save_flag=0
	while save_flag==0:
		density,count_map,rhototal_map,plot_coor=multi_lines(n_dis,total,neighborlist,boundary)
		if np.amax(count_map)<=5:
			f=plot_line(plot_coor,total,density,count_map,rhototal_map)
			save_flag = int(input("Save or not "))
			if save_flag!=0:
				pathlib.Path(os.getcwd()+"\\plot").mkdir(parents=True, exist_ok=True)
				f.savefig(os.getcwd()+'\\plot\\plot_total_'+str(save_flag)+'.jpeg')
	
				init_file_name=os.getcwd()+"\\plot\\large_total_"+str(save_flag)+".npz"
				print(init_file_name)
				np.savez(init_file_name, density,plot_coor,count_map,rhototal_map)
				break
def main():
	# demo_plot(2)
	global resolution,half_length
	resolution=31
	half_length=4.5
	n_label=5
	generate_sample(1,int(1e5),n_label,'train')
	generate_sample(1,int(2e4),n_label,'validation')
	generate_sample(1,int(2e4),n_label,'test')
	input_path=os.getcwd()+"\\input"
	conta_file(input_path,n_label)

main()