import zmq, sys
from PIL import Image
import cv2, json

context = zmq.Context()



socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

def send(img):
	#frame = cv2.imread(img)
	frame = Image.open(img)
	socket.send_pyobj(frame)

	message = socket.recv_json()
	#print(message["label"])
	#output_json = json.loads(message)
	return message

out = send(sys.argv[1])
print(out)
