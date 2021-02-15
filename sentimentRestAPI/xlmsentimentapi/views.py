#reference : https://www.youtube.com/watch?v=nN2Vp15AW5w

from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import json
from django.http import JsonResponse

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer
from tensorflow.keras.layers import Dense, Input


# Create your views here.

@api_view(["POST"])

def xlm_sentiment(data):
    try:
        
        modelpath = 'E:\\sentimentRestAPI\\sentimentxlm\\uploads\\model_checkpoint_0.h5'
        #modelpath = './model_checkpoint_0.h5'
        MODEL = 'jplu/tf-xlm-roberta-base'
        maxlen = 512

        def regular_encode(texts, tokenizer, maxlen=512):
            enc_di = tokenizer.batch_encode_plus(
                texts, 
                return_attention_masks=False, 
                return_token_type_ids=False,
                pad_to_max_length=True,
                max_length=maxlen
            )
            
            return np.array(enc_di['input_ids'])
            

        #from keras_radam import RAdam
        def build_model(transformer, loss='binary_crossentropy', max_len=512):
            input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
            sequence_output = transformer(input_word_ids)[0]
            cls_token = sequence_output[:, 0, :]
            #x = tf.keras.layers.Dropout(0.3)(cls_token)
            out = Dense(1, activation='sigmoid')(cls_token)
            
            
            model = Model(inputs=input_word_ids, outputs=out)
            #model.compile(RAdam( warmup_proportion=0.1, min_lr=1e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])
            
            return model
            
           
        strategy = tf.distribute.MirroredStrategy()


        tokenizer = AutoTokenizer.from_pretrained(MODEL) #, use_fast=False


        with strategy.scope():
            transformer_layer = transformers.TFXLMRobertaModel.from_pretrained(MODEL)
            model = build_model(transformer_layer, max_len=maxlen)
        model.summary()

        model.load_weights(modelpath)

        abc_series = json.loads(data.body)
        abc_series = pd.Series(abc_series)

        x_test1 = regular_encode(abc_series, 
                             tokenizer, maxlen=maxlen)


        test_dataset1 = (
            tf.data.Dataset
            .from_tensor_slices(x_test1)
            .batch(1)
        )

        pred = model.predict(test_dataset1, verbose=1)

        pred = float(pred)
        
        if(pred >= 0.5):
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        return JsonResponse("Sentiment = : "+sentiment+" predicted score  = "+str(pred),safe = False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
