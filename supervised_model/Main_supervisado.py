import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.layers as l
import matplotlib.pyplot as plt
# import tensorflow_addons as ta
from sklearn.model_selection import train_test_split

DEBUG = False

if DEBUG : print(tf.__version__)
fileName = './supervised_model/DATA.h5'
with open(fileName, 'rb') as r:
    DATA = pickle.load(r)

STATES, ACTIONS = DATA['s'], DATA['a']
if DEBUG : print(f'# States {len(STATES)} # Actions {len(ACTIONS)}')

gpus = False
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# capacity=3000
if gpus :
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            print("\n\nENTRE EN LAS GPUS IN PUSE SET MEMORY GROWTH TRUE")
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=capacity*0.8)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def build_model(input_shape,type="conv"):
    layers=[]
    if type=="conv":
        layers = [l.Conv1D(32,10,activation=gelu,strides=1,padding="same",input_shape= input_shape),
              l.Dropout(0.5),
              # ta.layers.Maxout(32),
            l.Conv1D(64,10,activation=gelu,strides=5),
            l.BatchNormalization(),
            l.Dropout(0.5),
            l.LocallyConnected1D(128,5,activation=gelu),
            l.BatchNormalization(),
            l.Flatten(),
            l.Dropout(0.4),
            l.Dense(7*7,activation="softmax")
            ]
    if type == "fc":
        layers = [l.InputLayer(input_shape,dtype=tf.float32),
                  l.Reshape((91,)),
                l.Dense(512,activation="tanh",input_shape=(91,)),
                l.BatchNormalization(),
                l.Dropout(0.5),
                l.Dense(512, activation="tanh"),
                l.BatchNormalization(),
                l.Dropout(0.5),
                l.Dense(7*7,activation="softmax")

        ]

    model = tf.keras.Sequential(layers)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy()])
    # model.build(input_shape=input_shape)
    if DEBUG : model.summary()
    return model

def arreglar_datos(x_pre,y_pre,mode=1):
    x_pos=[]
    y_pos=[]
    for i in range(len(x_pre)):

        temp_x =[]
        for elem in x_pre[i]:
            temp_x.extend(elem)
        if mode==1:
            x_pos.append(np.array(temp_x).reshape(-1,1))
        else:
            x_pos.append(np.array(temp_x))
        y_pos.append(np.array(y_pre[i]).ravel())
    return np.array(x_pos,dtype=np.int32),np.array(y_pos,dtype=np.int32)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value1,y1 = loss(model, inputs, targets, training=True)
  return loss_value1, tape.gradient([loss_value1],model.trainable_variables),y1

def crear_batch(x,y,batch_size):
    index = np.random.choice(range(len(x)),batch_size)
    return x[index,:],y[index,:]

def loss(model, x, y, training):
    loss_object1 = tf.keras.losses.CategoricalCrossentropy()
    y_pred = model(x,training)
    return loss_object1(y_true=y,y_pred=y_pred)

def evaluate_model(model,X,Y):
    n=len(Y)
    y_pred = model.predict(X)
    acuraccy_obej = tf.keras.metrics.CategoricalAccuracy()
    print("Accuracy in {} samples of test : {}".format(n,acuraccy_obej(Y,y_pred)))

def train_model(model,X,Y,log_name,model_name,batch_size=32,step_per_epoch=10,epochs=10):
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
    # train_dataset =tf.data.Dataset.from_tensor_slices((X,Y)).batch(batch_size=batch_size).repeat().shuffle(1000)
    print("Voy a empezar el entrenamiento")
    for epoch in range(epochs):

        epoch_loss_avg1 = tf.keras.metrics.Mean()

        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Training loop - using batches of 32

        for i in range(step_per_epoch):
            x,y = crear_batch(X,Y,batch_size)
            loss_value1, grads,y1 = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg1(loss_value1)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(y,y1)
            # print("Log_end: " + str(y2.numpy()))
        if epoch%10 == 0:
            model.save(model_name)
        # End epoch
        # train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results_start.append(epoch_accuracy_start.result())
        # train_accuracy_results_end.append(epoch_accuracy_end.result())

        # if epoch % 50 == 0:
        f=open(log_name,"a")
        print("Epoch {:03d}: Loss start : {:.3f}, Accuracyt: {:.3%} ".format(epoch,epoch_loss_avg1.result(),epoch_accuracy.result()))

        f.write("Epoch {:03d}: Loss start : {:.3f}, Accuracyt: {:.3%} ".format(epoch,epoch_loss_avg1.result(),epoch_accuracy.result()))
        # f.write("Log_end: " + str(y2.numpy()))
        f.close()

def train():
    x,y = arreglar_datos(STATES,ACTIONS,mode=1)
    bad_index = np.where(np.max(y,axis=1)== 0)
    x=np.delete(x,bad_index,0)
    y=np.delete(y,bad_index,0)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    model = build_model((x[0].shape[0],x[0].shape[1],),type="fc")


    #Callbacks

    model_callback = tf.keras.callbacks.ModelCheckpoint("models/DOMINATOR_e{epoch}-val_loss_{val_loss:.4f}.hdf5",save_best_only=True)
    reduce_learning = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=1e-8)
    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience = 15, verbose=1, mode='auto', restore_best_weights=True)

    import datetime
    t = datetime.datetime.now().time()
    log_name = "Entrenamiento_{}.txt".format(t)
    # train_model(model,X_train,y_train,log_name,"DOMINATOR.hdf5",step_per_epoch=20,epochs=100)
    #



    history = model.fit(X_train,y_train,batch_size=64,epochs=1000,verbose=2,callbacks=[model_callback,reduce_learning,early_callback],validation_split=0.1)
    # model.load_weights("models/DOMINATOR_e19-val_loss_2.6855.hdf5")
    cosas = model.evaluate(X_test,y_test,verbose=1)
    print(cosas)
    y_pred = model.predict(X_test)


    with open("y_pred","w+b")as f:
        pickle.dump(y_pred,f)
    with open("y_true","w+b")as f:
        pickle.dump(y_test,f)
    with open("x","w+b")as f:
        pickle.dump(X_test,f)
    indexes = np.random.randint(0,len(y_test),4)

    for i in indexes:
        real = y_test[i,:].reshape(7,7)
        predicted = y_pred[i,:].reshape(7,7)
        plt.figure()
        ax = plt.subplot(1,2,1)
        ax.set_title("y real")
        ax.matshow(real)
        ax = plt.subplot(1,2,2)
        ax.set_title("y predicho")
        ax.matshow(predicted)

        plt.savefig("comparación_{}.png".format(i))
    print("Accuracy en test: {}".format(tf.keras.metrics.CategoricalAccuracy()(y_test,y_pred)))

    # loss_ = history.history["loss"]
    # val_loss = history.history["val_loss"]

