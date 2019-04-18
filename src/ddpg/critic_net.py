import numpy as np
import tensorflow as tf
import math

TAU = 0.001
LEARNING_RATE = 0.001
BATCH_SIZE = 64


class CriticNet:
    """ Critic Q value model of the DDPG algorithm """

    def __init__(self, num_states, num_actions, dim_embed):
        self.dim_embed = dim_embed
        self.g = tf.Graph()
        self.freeze_embed = False
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # critic_q_model parameters:
            self.embed_c, self.embed_W_1, self.embed_B_1, self.embed_W_2, self.embed_B_2, self.W1_c, self.B1_c,\
                    self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c,\
                    self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net(
                    num_states, num_actions)

            # create target_q_model:
            self.t_embed_c, self.t_embed_W_1, self.t_embed_B_1, self.t_embed_W_2, self.t_embed_B_2, self.t_W1_c,\
                    self.t_B1_c, self.t_W2_c, self.t_W2_action_c, self.t_B2_c, self.t_W3_c, self.t_B3_c,\
                    self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net(
                    num_states, num_actions)

            self.q_value_in = tf.placeholder("float", [None, 1])  # supervisor
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c)
            self.l2_regularizer_loss = 0.0001 * \
                tf.reduce_sum(tf.pow(self.W2_c, 2)) + 0.0001 * tf.reduce_sum(tf.pow(self.B2_c, 2))
            self.cost = tf.pow(self.critic_q_model - self.q_value_in, 2) / BATCH_SIZE + \
                self.l2_regularizer_loss  # /tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
            self.freeze_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost, var_list=[self.W1_c, self.B1_c,
                self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c])

            # action gradient to be used in actor network:
            # self.action_gradients=tf.gradients(self.critic_q_model,self.critic_action_in)
            # from simple actor net:
            self.act_grad_v = tf.gradients(self.critic_q_model, self.embed_c)
            # this is just divided by batch size
            self.action_gradients = [self.act_grad_v[0] /
                                     tf.to_float(tf.shape(self.act_grad_v[0])[0])]
            # from simple actor net:
            self.check_fl = self.action_gradients

            # initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())

            self.update_target_critic_op = [
                self.t_embed_W_1.assign(TAU * self.embed_W_1 + (1 - TAU) * self.t_embed_W_1),
                self.t_embed_B_1.assign(TAU * self.embed_B_1 + (1 - TAU) * self.t_embed_B_1),
                self.t_embed_W_2.assign(TAU * self.embed_W_2 + (1 - TAU) * self.t_embed_W_2),
                self.t_embed_B_2.assign(TAU * self.embed_B_2 + (1 - TAU) * self.t_embed_B_2),
                self.t_W1_c.assign(TAU * self.W1_c + (1 - TAU) * self.t_W1_c),
                self.t_B1_c.assign(TAU * self.B1_c + (1 - TAU) * self.t_B1_c),
                self.t_W2_c.assign(TAU * self.W2_c + (1 - TAU) * self.t_W2_c),
                self.t_W2_action_c.assign(TAU * self.W2_action_c + (1 - TAU) * self.t_W2_action_c),
                self.t_B2_c.assign(TAU * self.B2_c + (1 - TAU) * self.t_B2_c),
                self.t_W3_c.assign(TAU * self.W3_c + (1 - TAU) * self.t_W3_c),
                self.t_B3_c.assign(TAU * self.B3_c + (1 - TAU) * self.t_B3_c)
            ]
            # To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_embed_W_1.assign(self.embed_W_1),
                self.t_embed_B_1.assign(self.embed_B_1),
                self.t_embed_W_2.assign(self.embed_W_2),
                self.t_embed_B_2.assign(self.embed_B_2),
                self.t_W1_c.assign(self.W1_c),
                self.t_B1_c.assign(self.B1_c),
                self.t_W2_c.assign(self.W2_c),
                self.t_W2_action_c.assign(self.W2_action_c),
                self.t_B2_c.assign(self.B2_c),
                self.t_W3_c.assign(self.W3_c),
                self.t_B3_c.assign(self.B3_c)
            ])

    def freeze(self):
        self.freeze_embed = True

    def unfreeze(self):
        self.freeze_embed = False

    def create_critic_net(self, num_states=4, num_actions=1):
        N_HIDDEN_0 = 100
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        DIM_EMBED = self.dim_embed
        critic_state_in = tf.placeholder("float", [None, num_states])
        critic_action_in = tf.placeholder("float", [None, num_actions])

        embed_W_1 = tf.Variable(tf.random_uniform(
            [num_actions, N_HIDDEN_0], -1 / math.sqrt(num_actions), 1 / math.sqrt(num_actions)))
        embed_B_1 = tf.Variable(tf.random_uniform(
            [N_HIDDEN_0], -1 / math.sqrt(num_actions), 1 / math.sqrt(num_actions)))
        embed_W_2 = tf.Variable(tf.random_uniform(
            [N_HIDDEN_0, DIM_EMBED], -1 / math.sqrt(N_HIDDEN_0), 1 / math.sqrt(N_HIDDEN_0)))
        embed_B_2 = tf.Variable(tf.random_uniform(
            [DIM_EMBED], -1 / math.sqrt(N_HIDDEN_0), 1 / math.sqrt(N_HIDDEN_0)))
        W1_c = tf.Variable(tf.random_uniform(
            [num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        B1_c = tf.Variable(tf.random_uniform(
            [N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        W2_c = tf.Variable(tf.random_uniform(
            [N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + DIM_EMBED), 1 / math.sqrt(N_HIDDEN_1 + DIM_EMBED)))
        W2_action_c = tf.Variable(tf.random_uniform(
            [DIM_EMBED, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + DIM_EMBED), 1 / math.sqrt(N_HIDDEN_1 + DIM_EMBED)))
        B2_c = tf.Variable(tf.random_uniform(
            [N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + DIM_EMBED), 1 / math.sqrt(N_HIDDEN_1 + DIM_EMBED)))
        W3_c = tf.Variable(tf.random_uniform([N_HIDDEN_2, 1], -0.003, 0.003))
        B3_c = tf.Variable(tf.random_uniform([1], -0.003, 0.003))
        
        embed_c = tf.matmul(tf.nn.tanh(tf.matmul(critic_action_in, embed_W_1) + embed_B_1), embed_W_2) + embed_B_2
        H1_c = tf.nn.softplus(tf.matmul(critic_state_in, W1_c) + B1_c)
        H2_c = tf.nn.tanh(tf.matmul(H1_c, W2_c) + tf.matmul(embed_c, W2_action_c) + B2_c)

        critic_q_model = tf.matmul(H2_c, W3_c) + B3_c

        return embed_c, embed_W_1, embed_B_1, embed_W_2, embed_B_2, W1_c, B1_c, W2_c, W2_action_c, B2_c, W3_c, B3_c, critic_q_model, critic_state_in, critic_action_in

    def embed(self):
        def scale(arr):
            return self.sess.run(self.t_embed_c, feed_dict={self.t_critic_action_in: arr})
        return scale

    #def embed_training(self):
    #    def scale(arr):
    #        return self.sess.run(self.embed_c, feed_dict={self.critic_action_in: arr})
    #    return scale

    def train_critic(self, state_t_batch, action_batch, y_i_batch):
        self.sess.run(self.optimizer if not self.freeze_embed else self.freeze_optimizer, feed_dict={
                      self.critic_state_in: state_t_batch, self.critic_action_in: action_batch, self.q_value_in: y_i_batch})

    def evaluate_target_critic(self, state_t_1, action_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1, self.t_embed_c: action_t_1})

    def evaluate_critic(self, state_1, action_1):
        return self.sess.run(self.critic_q_model, feed_dict={self.critic_state_in: state_1, self.critic_action_in: action_1})

    def compute_delQ_a(self, state_t, action_t):
        #        print '\n'
        #        print 'check grad number'
        #        ch= self.sess.run(self.check_fl, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})
        #        print len(ch)
        #        print len(ch[0])
        #        raw_input("Press Enter to continue...")
        return self.sess.run(self.action_gradients, feed_dict={self.critic_state_in: state_t, self.embed_c: action_t})

    def update_target_critic(self):
        self.sess.run(self.update_target_critic_op)
