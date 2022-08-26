from fileinput import filename
from model import KModel
import os
import sys
import tensorflow as tf
import tempfile
from parser import KParseArgs
import keras
from keras import optimizers
from keras import metrics
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from preprocessing import PreProcess
from prepare_data import DATA
import pickle
from graphs import KPlot


class KTrain():

    def __init__(self):
        return
    
    def compile_and_fit_model(self, model, x_train, y_train,x_test,y_test,optimizer, epochs, batch_size,save_model,loss='categorical_crossentropy',
    metrics=(["accuracy"]),verbose=1,output_dir='/tmp'):
     

        if optimizer == 'adam':
            opt = optimizers.Adam()
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=[metrics])

        # Configure for TensorBoard visualization
        # Reference: [Monitor progress of your Keras based neural network using TensorBoard](https://bit.ly/2C36EBJ)
        print("Writing TensorFlow events locally to %s\n" % output_dir)
        tensorboard = TensorBoard(log_dir=output_dir)

        #
        # fit the model: use part of the training data and use validation for unseen data
        #
        
        
        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose,
                            validation_data=(x_test, y_test),
                            callbacks=[tensorboard,EarlyStopping(monitor="val_loss",patience=5,
                           mode="auto",restore_best_weights=True)])

        

        return history

    


    def evaluate_model(self,model, x_test, y_test):
        """
        Evaluate the model with unseen and untrained data
        :param model:
        :return: results of probability
        """

        return model.evaluate(x_test, y_test)

    def get_loss(self, hist):
        loss = hist.history['loss']
        loss_val = loss[len(loss) - 1]
        return loss_val

    def get_acc(self, hist):
        acc = hist.history['accuracy']
        acc_value = acc[len(acc) - 1]

        return acc_value

    def get_validation_loss(self, hist):
        val_loss = hist.history['val_loss']
        val_loss_value = val_loss[len(val_loss) - 1]

        return val_loss_value

    def get_validation_acc(self, hist):
        val_acc = hist.history['val_accuracy']
        val_acc_value = val_acc[len(val_acc) - 1]

        return val_acc_value


    def print_metrics(self, hist):

        acc_value = self.get_acc(hist)
        loss_value = self.get_loss(hist)

        val_acc_value = self.get_validation_acc(hist)

        val_loss_value = self.get_validation_loss(hist)

        print("Final metrics: loss:%6.4f" % loss_value)
        print("Final metrics: accuracy=%6.4f" % acc_value)
        print("Final metrics: validation_loss:%6.4f" % val_loss_value)
        print("Final metrics: validation__accuracy:%6.4f" % val_acc_value)


    def get_directory_path(self, dir_name, create_dir=True):

        cwd = os.getcwd()
        dir = os.path.join(cwd, dir_name)
        if create_dir:
            if not os.path.exists(dir):
                os.mkdir(dir, mode=0o755)

        return dir

    def train_models(self, args):
        
        # Create TensorFlow Session
        sess = tf.compat.v1.InteractiveSession()

        # Configure output_dir
        output_dir = tempfile.mkdtemp()

        
        
        ktrain_cls = KTrain()
        
        data = DATA()
        preprocessing = PreProcess()
        kplot_cls = KPlot()
        

        

        df = data.data_for_model("train_data.csv")
        text = preprocessing.prepare_text(df["text"])
        target = df["label"]

        X_train,X_test,y_train,y_test = train_test_split(text,target,test_size=.2,stratify=target,random_state=42)
        
        x_train,x_test,tokenizer = preprocessing.make_sequences(X_train,X_test,args.max_len)

        
        

        y_train,y_test = preprocessing.prepare_targets(y_train,y_test)
        model_type=args.model_type
        image_dir = ktrain_cls.get_directory_path("images")
        model_dir = ktrain_cls.get_directory_path("models")
        if model_type == "base_line":
            graph_label_loss = 'Baseline Model: Training and Validation Loss'
            graph_label_acc = 'Baseline Model: Training and Validation Accuracy'
            graph_image_loss_png = os.path.join(image_dir,'baseline_loss.png')
            graph_image_acc_png = os.path.join(image_dir, 'baseline_accuracy.png')

        if model_type == "cnn":
            graph_label_loss = 'CNN: Training and Validation Loss'
            graph_label_acc = 'CNN Model: Training and Validation Accuracy'
            graph_image_loss_png = os.path.join(image_dir, 'cnn_loss.png')
            graph_image_acc_png = os.path.join(image_dir,'cnn_accuracy.png')

        if model_type == "lstm":
            graph_label_loss = 'LSTM: Training and Validation Loss'
            graph_label_acc = 'LSTM Model: Training and Validation Accuracy'
            graph_image_loss_png = os.path.join(image_dir, 'lstm_loss.png')
            graph_image_acc_png = os.path.join(image_dir,'lstm_accuracy.png')
        

        

        
       

   
        size = len(tokenizer.word_index) + 1
        print(size)
        embedding_dim = 50
        filepath = "glove.6B.50d.txt"
        word_index = tokenizer.word_index
        embeddding_matrix = preprocessing.create_embedding_matrix(filepath, word_index, embedding_dim,size)
        
        kmodel = KModel()
        print(model_type+" :")
        if model_type == "base_line":
            model = kmodel.baseline_model(args.embedding_dim,embeddding_matrix,args.num_classes,args.max_len,size)
        if model_type =="cnn":
            
            model = kmodel.model_cnn(args.embedding_dim,size,args.max_len,args.num_classes,embeddding_matrix)
        if model_type =="lstm":
            
            model = kmodel.model_lstm(size,args.embedding_dim,args.max_len,args.num_classes,embeddding_matrix)

        
        model.summary()
        history = ktrain_cls.compile_and_fit_model(model, x_train, y_train,x_test,y_test,optimizer = args.opt,batch_size = args.batch_size,save_model = args.save_model,
        epochs=args.epochs, loss=args.loss, output_dir=output_dir)

        figure_loss = kplot_cls.plot_loss_graph(history, graph_label_loss)
        figure_loss.savefig(graph_image_loss_png)
        figure_acc = kplot_cls.plot_accuracy_graph(history, graph_label_acc)
        figure_acc.savefig(graph_image_acc_png)

        if args.save_model==True:
            #model_dir = self.get_directory_path("models")
            dir = os.path.join('models', args.model_type)
            if not os.path.exists(dir):
               os.mkdir(dir, mode=0o755)

            model.save(dir+'/'+str(args.model_type)+'.h5')
            with open('models/'+str(args.model_type)+'/tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        ktrain_cls.print_metrics(history)
        results = ktrain_cls.evaluate_model(model, x_test, y_test)
        print("Average Probability Results:")
        print(results)

        

if __name__ == '__main__':
    #
    # main used for testing the functions
    #
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    print("model_type:", args.model_type)

    print("embedding dimension:", args.embedding_dim)
    
    print("epochs:", args.epochs)
    print("loss:", args.loss)
    print("number of classes:", args.num_classes)
    print("max_len:", args.max_len)

    KTrain().train_models(args)
