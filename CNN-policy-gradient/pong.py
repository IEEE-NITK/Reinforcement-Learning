################## Pong game: Policy Gradient using CNN ##########################

import numpy as np
import pickle
import gym
import tensorflow as tf
import time
import os

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float)


def discount_rewards(r):
  gamma = 0.99
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def load_model(path):
  model = pickle.load(open(path, 'rb'))
  return model['W1'].T, model['W2'].reshape((model['W2'].size,-1))

def make_network(pixels_num):

  pixels = tf.placeholder(dtype=tf.float32, shape=(None, pixels_num, pixels_num, 1))    
  actions = tf.placeholder(dtype=tf.float32, shape=(None,1))
  rewards = tf.placeholder(dtype=tf.float32, shape=(None,1))

  with tf.variable_scope('policy'):
    conv1 = tf.layers.conv2d(inputs=pixels, filters=200, kernel_size=[80,80], padding="valid", activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)
    
    #conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=4)
    
    pool1_flat = tf.reshape(conv1, [-1, 1 * 1 * 200])
    #dense = tf.layers.dense(inputs=conv1, units=32, activation=tf.nn.relu,  kernel_initializer = tf.contrib.layers.xavier_initializer())
    dropout = tf.layers.dropout(inputs=pool1_flat, rate=0.3)
    
    
    
#     hidden = tf.layers.dense(pixels, hidden_units, activation=tf.nn.relu,\
#             kernel_initializer = tf.contrib.layers.xavier_initializer())
    logits = tf.layers.dense(dropout, 1, activation=None,\
            kernel_initializer = tf.contrib.layers.xavier_initializer())

    out = tf.sigmoid(logits, name="sigmoid")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=actions, logits=logits, name="cross_entropy")
    loss = tf.reduce_sum(tf.multiply(rewards, cross_entropy, name="rewards"))
    
  # lr=1e-4
  lr=1e-4
  decay_rate=0.99
  opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)
  # opt = tf.train.AdamOptimizer(lr).minimize(loss)

  tf.summary.histogram("conv1_out", conv1)
  #tf.summary.histogram("pool1_out", pool1)
  #tf.summary.histogram("conv2_out", conv2)
  #tf.summary.histogram("pool2_out", pool2)
  tf.summary.histogram("pool1_flat_out", pool1_flat)
  #tf.summary.histogram("dense_out", dense)
  tf.summary.histogram("dropout_out", dropout)
  tf.summary.histogram("logits_out", logits)
  tf.summary.histogram("prob_out", out)
  merged = tf.summary.merge_all()

  # grads = tf.gradients(loss, [hidden_w, logit_w])
  # return pixels, actions, rewards, out, opt, merged, grads
  return pixels, actions, rewards, out, opt, merged


pixels_num = 80
batch_size = 3 #5

tf.reset_default_graph()
pix_ph, action_ph, reward_ph, out_sym, opt_sym, merged_sym = make_network(pixels_num)

resume = True              # change resume to True if you want to load a saved model and continue training
# resume = False 
render = True              # change render to True if you want the game to be displayed

sess = tf.Session()
saver = tf.train.Saver()
writer = tf.summary.FileWriter('./log/train', sess.graph)

if resume:
  saver.restore(sess, tf.train.latest_checkpoint('./log/checkpoints'))
else:
  sess.run(tf.global_variables_initializer())

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs = []
ys = []
ws = []
ep_ws = []
batch_ws = []
step = pickle.load(open('./log/step.p', 'rb')) if resume and os.path.exists('./log/step.p') else 0
episode_number = 2655 + (step - 401) * batch_size
reward_mean = -21.0

while True:
  if render: env.render()
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros((pixels_num,pixels_num))
  prev_x = cur_x

  tf_probs = sess.run(out_sym, feed_dict={pix_ph:x.reshape((-1, pixels_num,pixels_num, 1))})
  y = 1 if np.random.uniform() < tf_probs[0,0] else 0
  action = 2 + y
  observation, reward, done, info = env.step(action)

  xs.append(x)

  ys.append(y)
  ep_ws.append(reward)

  if done:
    episode_number += 1
    discounted_epr = discount_rewards(ep_ws)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    batch_ws += discounted_epr.tolist()

    reward_mean = 0.99*reward_mean+(1-0.99)*(sum(ep_ws))
    rs_sum = tf.Summary(value=[tf.Summary.Value(tag="running_reward", simple_value=reward_mean)])
    writer.add_summary(rs_sum, global_step=episode_number)
    ep_ws = []
    if reward_mean > 5.0:
        break

    if episode_number % batch_size == 0:
        step += 1
        xs = np.array(xs)

        exs = np.stack(xs)
        exs = exs.reshape((-1,pixels_num,pixels_num,1))
        eys = np.stack(ys)
        eys = eys.reshape((-1,1))
        ews = np.stack(batch_ws)
        ews = ews.reshape((-1,1))

        frame_size = len(xs)
        xs = []
        ys = []
        batch_ws = []

        tf_opt, tf_summary = sess.run([opt_sym, merged_sym], feed_dict={pix_ph:exs,action_ph:eys,reward_ph:ews})
        saver.save(sess, "./log/checkpoints/pg_{}.ckpt".format(step))
        writer.add_summary(tf_summary, step)
        print("datetime: {}, episode: {}, update step: {}, frame size: {}, reward: {}".\
                format(time.strftime('%X %x %Z'), episode_number, step, frame_size, reward_mean))
        

        fp = open('./log/step.p', 'wb')
        pickle.dump(step, fp)
        fp.close()

    observation = env.reset()            
    if render: env.render()

env.close()