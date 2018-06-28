import os
import numpy as np
import tensorflow as tf
import math as math

def process_npz(count,rhototal):
	global n_output
	Y_Train=np.zeros((len(count),n_output))
	for elem in range(len(count)):
		Y_Train[elem,int(count[elem]-1)]=1
	count_y=Y_Train
	count_y[:,-1]=rhototal[:]
	return count_y
	
def read_input(path):
	init_file_name=path+"\\crop_1_5_total_comb.npz"
	if os.path.isfile(init_file_name):
		init_file=np.load(init_file_name)
		
		density_x=init_file['density']
		count_y=process_npz(init_file['count'],init_file['rhototal'])
		
		t_density_x=(init_file['t_density'])
		t_count_y=process_npz(init_file['t_count'],init_file['t_rhototal'])
		
		init_file.close()
		return density_x,count_y,t_density_x,t_count_y
	else:
		print('Missing initialize data!')

def same_conv_ba_r(prev,n,k,t_flag):
	initial_weight=tf.contrib.layers.xavier_initializer(uniform=False,seed=31423)
	initial_bias=tf.constant_initializer(0.1)
	next=tf.contrib.layers.conv2d(prev, n, kernel_size=(k,k),
								weights_initializer=initial_weight,
								biases_initializer=initial_bias,
								padding='SAME',
								activation_fn=None)
	next=tf.layers.batch_normalization(next,training=t_flag)
	next=tf.nn.relu(next)
	return next

def valid_conv_ba_r(prev,n,k,t_flag):
	initial_weight=tf.contrib.layers.xavier_initializer(uniform=False,seed=31423)
	initial_bias=tf.constant_initializer(0.1)
	next=tf.contrib.layers.conv2d(prev, n, kernel_size=(k,k),
								weights_initializer=initial_weight,
								biases_initializer=initial_bias,
								padding='valid',
								activation_fn=None)
	next=tf.layers.batch_normalization(next,training=t_flag)
	next=tf.nn.relu(next)
	return next

def build_ConvNets(ratio,regression_type):
	global imagesize,n_output	
	tf.reset_default_graph()
	global_steps = tf.Variable(0, trainable=False)

	X=tf.placeholder(tf.float32,[None,imagesize,imagesize,3])

	Y=tf.placeholder(tf.float32, [None, n_output])
	Dropout_s = tf.placeholder(tf.float32)
	training_flag = tf.placeholder(tf.bool)

	initial_weight=tf.contrib.layers.xavier_initializer(uniform=False,seed=31423)
	initial_bias=tf.constant_initializer(0.1)

	
	#layers
	X1=tf.layers.batch_normalization(X,training=training_flag)

	c1_1=same_conv_ba_r(X1,25,3,training_flag)
	c1_2=same_conv_ba_r(c1_1,25,3,training_flag)



	c2_1=same_conv_ba_r(X1,10,1,training_flag)
	c2_2=same_conv_ba_r(c2_1,25,3,training_flag)
								
	c3_1=same_conv_ba_r(X1,25,1,training_flag)							
	c3_2=tf.contrib.layers.max_pool2d(c3_1, kernel_size=(3,3),stride=1,
									padding='SAME')

	with tf.name_scope('concat'):
		conta1=tf.concat([c1_2, c2_2, c3_2],3)

	c4=same_conv_ba_r(conta1,30,3,training_flag)

	c5=same_conv_ba_r(c4,35,3,training_flag)

	c6=same_conv_ba_r(c5,40,3,training_flag)

	c7=same_conv_ba_r(c6,45,3,training_flag)

	c8=same_conv_ba_r(c7,50,3,training_flag)

	c9=same_conv_ba_r(c8,55,3,training_flag)

	c10=same_conv_ba_r(c9,60,3,training_flag)

	c11=same_conv_ba_r(c10,65,3,training_flag)

	c12=valid_conv_ba_r(c11,70,2,training_flag)
									
	c13=valid_conv_ba_r(c12,75,2,training_flag)

	c14=valid_conv_ba_r(c13,80,2,training_flag)

	c15=valid_conv_ba_r(c14,85,2,training_flag)

	c16=valid_conv_ba_r(c15,90,2,training_flag)

	c17=valid_conv_ba_r(c16,95,2,training_flag)

	c17=tf.contrib.layers.dropout(c17,keep_prob=Dropout_s)			

	F=tf.contrib.layers.flatten(c17)

	F1=tf.contrib.layers.fully_connected(F,128,
											weights_initializer=initial_weight,
											activation_fn=None,
											biases_initializer=initial_bias)
	F1=tf.layers.batch_normalization(F1,training=training_flag)													
	F1=tf.nn.relu(F1)
												
	DF1=tf.contrib.layers.dropout(F1,keep_prob=Dropout_s)
		
	Yhat_1=tf.contrib.layers.fully_connected(DF1,(n_output-1),
											weights_initializer=initial_weight,
											biases_initializer=initial_bias,
											activation_fn=None)

	Yhat_2=tf.contrib.layers.fully_connected(DF1,1,
											weights_initializer=initial_weight,
											biases_initializer=initial_bias,
											activation_fn=tf.nn.relu)
	Yhat=tf.concat([Yhat_1, Yhat_2],1)
										
	#loss
	with tf.name_scope('loss'):	
		correct_label=tf.argmax(Y[:,:-1], 1)
		loss_n=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_label,logits=Yhat[:,:-1]))
		if regression_type=='mse':
			loss_rho=tf.reduce_mean(tf.losses.mean_squared_error(labels=Y[:,-1],predictions=Yhat[:,-1]))
		elif regression_type=='log':
			loss_rho=tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.log(Y[:,-1]+1),predictions=tf.log(Yhat[:,-1]+1)))
		loss=ratio*loss_n+(1-ratio)*loss_rho

	# tf.summary.scalar('/loss',loss)

	with tf.name_scope('accuracy'):
		predict_op = tf.argmax(Yhat[:,:-1], 1)
		correct_prediction = tf.equal(predict_op, correct_label)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#RMSLE
		accuracy_rhototal=tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Yhat[:,-1],Y[:,-1]),Y[:,-1])))

	# tf.summary.scalar('/accuracy',accuracy)

	with tf.name_scope('adam_optimizer'):	
		initial_learning_rate = 1e-3
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			adam_opt=tf.train.AdamOptimizer(initial_learning_rate)
			opt=adam_opt.minimize(loss,global_steps)

	saver=tf.train.Saver()
	return global_steps,loss_rho,accuracy,accuracy_rhototal,X,Y,Dropout_s,training_flag,opt

def train(density_x,count_y,t_density_x,t_count_y,global_steps,loss_rho,accuracy,accuracy_rhototal,X,Y,Dropout_s,training_flag,opt,current_dir):
	train_ls,test_ls,train_acc_ls,test_acc_ls,train_acc_rhototal_ls,test_acc_rhototal_ls=[],[],[],[],[],[]
	record_best_acc=0.42
	
	train_size=int(100000*(n_output-1))
	test_size=(20000*(n_output-1))
	batch_size_set=64
	training_iters=math.ceil(train_size/batch_size_set)*60+1
	datasave_step=math.ceil(train_size/batch_size_set)*5
	display_step =math.ceil(train_size/batch_size_set)
	x_batch_shuffle,y_batch_shuffle=tf.train.shuffle_batch([density_x,count_y],
															enqueue_many=True, batch_size=batch_size_set, 
															capacity=train_size*2, min_after_dequeue=1000,
															allow_smaller_final_batch=True)
	t_x_batch,t_y_batch=tf.train.batch([t_density_x,t_count_y],
										enqueue_many=True, batch_size=int(test_size*0.001), 
										allow_smaller_final_batch=True)	
	x_batch,y_batch=tf.train.batch([density_x,count_y],
									enqueue_many=True, batch_size=batch_size_set, 
									allow_smaller_final_batch=True)	
									
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		tf.global_variables_initializer().run()
		print(train_size)
		print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]), " variables to train~")
		
		step = sess.run(global_steps)
		fraction=math.ceil(train_size/batch_size_set)
		while step < training_iters:
			
			if step % display_step==0:
				print(step)
				if step %int(1*display_step)==0:
					train_loss=0
					train_acc=0
					train_acc_rhototal=0
					for i in range(fraction):
						X_batch,Y_batch=sess.run([x_batch,y_batch])
						batch_loss,batch_acc,batch_acc_rhototal=sess.run([loss_rho,accuracy,accuracy_rhototal],feed_dict={X:X_batch,Y:Y_batch,Dropout_s:1,training_flag:False})
						train_loss+=batch_loss
						train_acc+=batch_acc
						train_acc_rhototal+=batch_acc_rhototal
					train_loss/=float(fraction)
					train_acc/=float(fraction)
					train_acc_rhototal/=float(fraction)
				print("Epoch ",int(step/math.ceil(train_size/batch_size_set)))

				test_loss=0
				test_acc=0
				test_acc_rhototal=0
				for i in range(1000):
					t_X_batch,t_Y_batch=sess.run([t_x_batch,t_y_batch])
					test_loss_p,test_acc_p,test_acc_rhototal_p = sess.run([loss_rho,accuracy,accuracy_rhototal],feed_dict={X:t_X_batch,Y:t_Y_batch,Dropout_s:1,training_flag:False})
					test_loss+=test_loss_p
					test_acc+=test_acc_p
					test_acc_rhototal+=test_acc_rhototal_p
				test_loss/=1000.
				test_acc/=1000.
				test_acc_rhototal/=1000.
				
				train_ls.append(train_loss)
				test_ls.append(test_loss)
				train_acc_ls.append(train_acc)
				test_acc_ls.append(test_acc)
				train_acc_rhototal_ls.append(train_acc_rhototal)
				test_acc_rhototal_ls.append(test_acc_rhototal)
				
				if test_acc_rhototal<record_best_acc:
					save_path = saver.save(sess, current_dir+"/tmp/model.ckpt")
					record_best_acc=test_acc_rhototal
					print("Model saved in file: %s" % save_path)
						
				print("tr_loss : %.4f" %(train_loss),"te_loss : %.4f" % (test_loss),
						"tr_acc : %.2f" %(train_acc),"te_acc : %.4f" %(test_acc),
						"tr_acc_t : %.2f" %(train_acc_rhototal),"te_acc_t : %.4f" %(test_acc_rhototal))
									
				if step % datasave_step==0 and step>0:
					ls_file_name=current_dir+"/ls_log.npz"
					train_ls_array=np.asarray(train_ls)
					test_ls_array=np.asarray(test_ls)
					train_acc_ls_array=np.asarray(train_acc_ls)
					test_acc_ls_array=np.asarray(test_acc_ls)
					train_acc_rhototal_ls_array=np.asarray(train_acc_rhototal_ls)
					test_acc_rhototal_ls_array=np.asarray(test_acc_rhototal_ls)

					np.savez(ls_file_name, train_ls=train_ls_array,test_ls=test_ls_array, 
								train_acc_ls=train_acc_ls_array,test_acc_ls=test_acc_ls_array,
								train_acc_rhototal_ls=train_acc_rhototal_ls_array,
								test_acc_rhototal_ls=test_acc_rhototal_ls_array)
		
			X_batch,Y_batch=sess.run([x_batch_shuffle,y_batch_shuffle])
			_,step=sess.run([opt,global_steps],feed_dict={X:X_batch,Y:Y_batch,Dropout_s:0.5,training_flag:True})		
		print("Optimization Finished!")
		coord.request_stop()
		coord.join(threads)
	#write acc to file
	ls_file_name=current_dir+"/ls_log.npz"
	train_ls_array=np.asarray(train_ls)
	test_ls_array=np.asarray(test_ls)
	train_acc_ls_array=np.asarray(train_acc_ls)
	test_acc_ls_array=np.asarray(test_acc_ls)

	np.savez(ls_file_name, train_ls=train_ls_array,test_ls=test_ls_array, 
						train_acc_ls=train_acc_ls_array,test_acc_ls=test_acc_ls_array,
						train_acc_rhototal_ls=train_acc_rhototal_ls_array,
						test_acc_rhototal_ls=test_acc_rhototal_ls_array)
def main():
	global imagesize,n_output
	imagesize=7
	n_output=(5+1)						
	current_dir=os.getcwd()
	density_x,count_y,t_density_x,t_count_y=read_input(current_dir+"\\input")
	
	global_steps,loss_rho,accuracy,accuracy_rhototal,X,Y,Dropout_s,training_flag,opt=build_ConvNets(0.5,'log')
	train(density_x,count_y,t_density_x,t_count_y,global_steps,loss_rho,accuracy,accuracy_rhototal,X,Y,Dropout_s,training_flag,opt,current_dir)
	
main()