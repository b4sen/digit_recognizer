import pickle
import cv2
PATH = '../../data/'
x_train = pickle.load(open(f'{PATH}x_train','rb'))
x_test = pickle.load(open(f'{PATH}x_test','rb'))

y_train = pickle.load(open(f'{PATH}y_train','rb'))
y_test = pickle.load(open(f'{PATH}y_test','rb'))


def process_img(img):
	blur = cv2.blur(img, (2,2))
	_, thresh = cv2.threshold(blur, 127,255, cv2.THRESH_BINARY)
	return thresh

def prepare_data(x,y):
	for img, lab in zip(x,y):
		yield (process_img(img), lab)


train = list(prepare_data(x_train, y_train))
test = list(prepare_data(x_test, y_test))

pickle.dump(train, open(f'{PATH}train', 'wb'))
pickle.dump(test, open(f'{PATH}test', 'wb'))