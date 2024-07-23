## To install required packages
pip3 install --requirement requirement.txt

## run the custom AQuA model on an Image
python3 inception_1block_classifier_1img.py [trained_model_path] [image_path]

*Example*

python3 inception_1block_classifier_1img.py trained_models/inception_1block_classifier_v1 test_imgs/2875_079217.jpg

python3 vgg_3blocks_classifier_testing_1img.py trained_models/vgg_classifier_freeze_v4 test_imgs/2875_079217.jpg

*output*

[image_path] [predicted_output_value] [predicted_class_label] [output_confidence]

## Augment input image with (sharp,color,contrast,bright) tuple value
python3 augment_img_tuples.py [input_image] [sharpness_value] [color_value] [contrast_value] [brightness_value] [output_directory]

*generated output file* will be stored at output_directory/

## How to serve the custom AQuA classifier model (Inception backbone) 
_Step 1: Start the server_

python3 inception_1block_classifier_server.py [trained_model_path]

python3 vgg_3blocks_classifier_testing_server.py [trained_model_path]

#for VGG

python3 vgg_3blocks_classifier_testing_server.py trained_models/vgg_classifier_freeze_v4

_Step 2: Request from localhost_

python3 zmq_client.py [image_path]

*zmq_client.py* contains *send()* method/API which returns top-1 prediction in JSON format. It contains label and confidence score
