import gym
import itertools
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

env=gym.make('CartPole-v0')
env.nS=4
env.nA=2

first=True
gamma=0.99
epsilon=np.finfo(np.float32).eps.item()
huber_loss=tf.keras.losses.Huber()
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
inputs=layers.Input(shape=(env.nS,))
common=layers.Dense(128,activation='relu')(inputs)
pi_head=layers.Dense(env.nA,activation='softmax')(common)
v_head=layers.Dense(1)(common)
model=keras.Model(inputs=inputs,outputs=[pi_head,v_head])

running_reward=0
step_limit=10_000
for episode in itertools.count():
    s,episode_reward=env.reset(),0
    P,V,R=[],[],[]
    with tf.GradientTape() as tape:
        for step in range(1,step_limit):
            s=tf.convert_to_tensor(s)
            s=tf.expand_dims(s,0)
            p,v=model(s)
            V.append(v[0,0])
            a=np.random.choice(env.nA,p=np.squeeze(p))
            P.append(tf.math.log(p[0,a]))
            s,r,d,i=env.step(a)
            episode_reward+=r
            R.append(r)
            if d:
                break
        running_reward=0.95*running_reward+0.05*episode_reward
        RET,cur=[],0
        for r in reversed(R):
            cur=r+gamma*cur
            RET.insert(0,cur)
        RET=np.array(RET)
        RET=(RET-np.mean(RET))/(np.std(RET)+epsilon)
        RET=RET.tolist()
        loss_p,loss_v=[],[]
        for log_p,v,ret in zip(P,V,RET):
            loss=ret-v
            loss_p.append(-log_p*loss)
            ret,v=tf.expand_dims(ret,0),tf.expand_dims(v,0)
            loss_v.append(huber_loss(ret,v))
        loss_ac=sum(loss_p)+sum(loss_v)
        grads=tape.gradient(loss_ac,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
    if episode%10==0 and episode>0:
        print(f'episode {episode}, running reward: {running_reward:.2f}')
    if running_reward > 195:
        print(f'solved on episode {episode}.')
        break
