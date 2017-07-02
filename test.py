import tensorflow as tf
import numpy as np
import math
import data
from PIL import Image




def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)



# Parameters
learning_rate = 0.0025
training_epochs = 15
batch_size = 100
display_step = 1

HIDDEN_NODES = 50

iteration= 4000;


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
o_color =data.readPalleteColor("body_pallete.png")
t1_color =data.readPalleteColor("body_pallete_02.png")
t2_color =data.readPalleteColor("body_pallete_03.png")
t3_color =data.readPalleteColor("body_pallete_04.png")

#xlen  줄



o_m_color = o_color[2]


x_arr= [


]
y_data=[

]

#강한색, 대상위치색,   새로운 강한색 -> 새로운 위치색


def fill_data(t_color, o_color,x_arr,y_data ):

     for idx, c in enumerate(t_color):
        a = []
        a.append(o_color[2])
        a.append(o_color[idx])
        a.append(t_color[2])
        line = np.array(a)
        line = line.flatten()
        x_arr.append(line)
        y_data.append(c)


fill_data(o_color,o_color,x_arr,y_data)
fill_data(t1_color,o_color,x_arr,y_data)
fill_data(t2_color,o_color,x_arr,y_data)
fill_data(t3_color,o_color,x_arr,y_data)


gen_arr=[]
def make_genData(o_color,new_m_color):
    for idx, c in enumerate(o_color):
        a = []
        a.append(o_color[2])
        a.append(o_color[idx])
        a.append(new_m_color)
        line = np.array(a)
        line = line.flatten()
        gen_arr.append(line)

new_m_color=(255,0,0)
make_genData(o_color, new_m_color)
gen_data=np.array(gen_arr )


x_data = np.array(x_arr)


x_len=len(x_data)



W1= tf.get_variable("W1", shape=[9,HIDDEN_NODES], initializer=xavier_init(9,HIDDEN_NODES))
W2= tf.get_variable("W2", shape=[HIDDEN_NODES,HIDDEN_NODES], initializer= xavier_init(HIDDEN_NODES,HIDDEN_NODES))
W3= tf.get_variable("W3", shape=[HIDDEN_NODES,3], initializer= xavier_init(HIDDEN_NODES,3))

L1 = tf.nn.relu(tf.matmul(X,W1))
L2 = tf.nn.relu(tf.matmul(L1,W2))
L3= tf.matmul(L2,W3)



cost = tf.reduce_mean(tf.square(L3 - y_data))

a= tf.Variable(learning_rate)
optimizer= tf.train.AdamOptimizer(a)
train =optimizer.minimize(cost)

init =tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    feed_dict = {X: x_data, Y: y_data}

    for step in range(iteration):

        _,out_accuracy=sess.run([train,cost], feed_dict= feed_dict )
        print(step, out_accuracy)

    output= sess.run([L3],feed_dict= { X:gen_data})


    print("cost", cost.eval(feed_dict))
    print("gen",output)

    for case_idx,result in enumerate(output) :
        img = Image.new('RGBA', (256, 1))
        pix = img.load()
        for x,color in enumerate(result) :
            r= min(255, max(0, round(color[0])))
            g=min(255, max(0, round(color[1])))
            b= min(255, max(0, round(color[2])))

            c=(  r,g,b,255)
            print("c",c)
            pix[x,0]= c

        img.save("o"+str(case_idx) + ".png")


