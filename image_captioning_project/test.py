import My_Projet as ei
import SceneDesc
import test_mod as tm
import sys

def text(imgs):
    #On charge le modèle VGG
    encode = ei.model_gen()
    
    #on charge le modèle RNN
    sd = SceneDesc.scenedesc()
    model = sd.create_model(ret_model = True)
    
    #On charge les poids qui vont avec
    weight = 'RNN_Train_weights/Weights.h5'
    model.load_weights(weight)
    
    #Où se situent nos images
    path = "Data/Images/"
    
    if isinstance(imgs, list): #si nous avons une liste d'imagess
        encoded_images = [(img, ei.encodings(encode, path + img)) for img in imgs]
        image_captions = [(img, tm.generate_captions(sd, model, encoding, beam_size=3)) for img, encoding in encoded_images]
        
    else: #Si nous avons une image unique
        image_path = path + imgs
        encoded_image = ei.encodings(encode, image_path)
        image_captions = (imgs, tm.generate_captions(sd, model, encoded_image, beam_size=3))
        
    print(image_captions)

if __name__ == '__main__':
	image = str(sys.argv[1])
	text(image)
