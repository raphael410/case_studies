import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Concatenate, concatenate #Merge
from keras.preprocessing import image, sequence
import pickle

EMBEDDING_DIM = 128

class scenedesc():
    def __init__(self):
        self.vocab_size = None
        self.no_samples = None
        self.max_length = None
        self.index_word = None
        self.word_index = None
        self.image_encodings = pickle.load(open("Output_step1/image_encodings.p", "rb"))
        self.captions = None
        self.img_id = None
        self.values()
        
    def values(self):
        dataframe = pd.read_csv('Data/Text/trainimgs.txt', delimiter=',', header=None)
        self.captions = []
        self.img_id = []
        self.no_samples=0
        
        for i in range(len(dataframe)):
            self.captions.append(dataframe.iloc[i,1])
            self.img_id.append(dataframe.iloc[i,0])
            
        tokens = [] #on split tous les captions en mot et on stock ici une liste de liste
        for caption in self.captions:
            self.no_samples+=len(caption.split())-1
            tokens.append(caption.split())

        vocab = [] #tous les mots rencontrés dans les captions
        for token in tokens:
            vocab.extend(token)
            
        vocab = list(set(vocab)) #On prend l'unicité (on passe par un set qui vire les doublons)
        self.vocab_size = len(vocab)

        caption_length = [len(caption.split()) for caption in self.captions] #list des longueurs des captions
        self.max_length = max(caption_length) #la plus grande, va servir d'input à notre modèle
        self.word_index = {} #dictionnaire mot : index
        self.index_word = {} #dictionnaire index : mot
        for i, word in enumerate(vocab):
            self.word_index[word]=i
            self.index_word[i]=word

    def data_process(self, batch_size):
        partial_captions = []
        next_words = []
        images = []
        total_count = 0
        while 1:
            image_counter = -1
            for caption in self.captions:
                image_counter+=1
                current_image = self.image_encodings[self.img_id[image_counter]]
                for i in range(len(caption.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in caption.split()[:i+1]]
                    partial_captions.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[caption.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_captions = sequence.pad_sequences(partial_captions, maxlen=self.max_length, padding='post')
                        total_count = 0
                        
                        yield [[images, partial_captions], next_words]
                        partial_captions = []
                        next_words = []
                        images = []


    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        image_input = Input(shape=[4096], dtype='float', name='image_input')
        image_model = Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu')(image_input)
        image_out = RepeatVector(self.max_length)(image_model)
        
        lang_input = Input(shape=(self.max_length,), dtype='int32', name='lang_input')
        lang_model = Embedding(self.vocab_size, 256, input_length=self.max_length)(lang_input)
        lang_model = LSTM(256, return_sequences=True)(lang_model)
        lang_out = TimeDistributed(Dense(EMBEDDING_DIM))(lang_model)
        
        final_model = concatenate([image_out, lang_out])
        final_model = LSTM(1000, return_sequences=False)(final_model)
        final_out = Dense(self.vocab_size, activation='softmax')(final_model)
        
        model = Model(inputs=[image_input, lang_input], outputs=[final_out])
        
        if(ret_model==True):
            return model
        
        print("Model Created")
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])
        
        return model

    def get_word(self,index):
        return self.index_word[index]

        
