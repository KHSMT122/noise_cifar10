from keras.datasets import cifar10
from keras.models import model_from_json
from keras.utils import np_utils

#cifar10をダウンロード
(_,_),(x_test_img,y_test_img)=cifar10.load_data()

#画像を0-1の範囲で正規化
x_test=x_test_img.astype('float32')/255.0

#正解ラベルをOne-Hot表現に変換
y_test=np_utils.to_categorical(y_test_img,10)

#学習済みのモデルと重みを読み込む
json_string=open('cifar10_cnn.json').read()
model=model_from_json(json_string)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('cifar10_cnn.h5')

#モデルを表示
model.summary()

#評価
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

#predict_classesで画像のクラスを予想する
img_pred=model.predict_classes(x_test)
