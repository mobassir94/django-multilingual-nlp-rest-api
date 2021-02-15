# django-multilingual-nlp-rest-api
A simple rest api using django for crosslingual sentiment analysis/toxic comment classification (NLP)

# How to install?

step 1 : install python 3.8.5,open powershell inside sentimentRestAPI folder and then,

step 2 : pip install virtualenv==20.2.2

step 3 : cd to sentimentRestAPI folder and do, virtualenv venv

step 4 : cd venv

step 5 : Scripts\activate

step 6 : then cd back to sentimentRestAPI  using (cd ..)

step 7 : now, install requirements.txt using this command  pip3 install -r requirements.txt

step 8 : cd sentimentxlm

step 9 : python manage.py runserver

step 10 : open postman and post url should be : http://127.0.0.1:8000/xlm_sentiment/ and inside body,pass message like this : {"data":"vhai khub valo manush"} and then hit the send button (for more clarification check the postman.png image inside sentimentxlm folder)


notes : 

1. make sure your trained model's weight file is inside sentimentRestAPI->sentimentxlm->uploads folder(i can't upload it here as it is hugee)
2. first api call can take time as it will download the gigantic pretrained model from internet(it depends on your internet speed and pc config)
3. you can use the weight file created in this task [Romanic-Bangla-Murad-Takla-Sentiment-Analysis](https://github.com/mobassir94/Romanic-Bangla-Murad-Takla-Sentiment-Analysis) for testing this rest api
