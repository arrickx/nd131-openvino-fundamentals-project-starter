"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
 
import os
import sys
import time
import socket
import json
import cv2
import time
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client() 
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)  
    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1) 
            count += 1
    return frame, count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold 

    ### TODO: Load the model through `infer_network` ###
    model=args.model
    video_input=args.input    
    cpu_extension=args.cpu_extension
    device=args.device
    infer_network.load_model(model, device, cpu_extension)
    net_input_shape = infer_network.get_input_shape()  

    ### TODO: Handle the input stream ###
    image_flag = False
    
    # If input is CAM 
    if video_input == 'CAM':
        input_stream = 0 
    # If input is image
    elif video_input.endswith('.jpg') or args.input.endswith('.bmp') :
        image_flag = True
        input_stream = video_input
    # input is video file
    else:
        input_stream = video_input
        assert os.path.isfile(video_input), " Video file is not found"
    
    if not image_flag: 
        cap = cv2.VideoCapture(video_input)
        cap.open(video_input) 
        
    width = int(cap.get(3))
    height = int(cap.get(4))
    duration,total_count,current_count,last_count = 0,0,0,0
    threshold_value = 2  

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read() 
        if not flag:
            break
        key_pressed = cv2.waitKey(60) 

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape) 
        infer_timer = time.time()

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)  

        ### TODO: Wait for the result ### 
        if infer_network.wait() == 0:
            inferece_time = time.time() - infer_timer  
            ### TODO: Get the results of the inference request ###  
            result = infer_network.get_output() 
            ### TODO: Extract any desired stats from the results ### 
            frame, count = draw_boxes(frame, result, args, width, height)   
            ### TODO: Calculate and send relevant information on ###   
            current_count = count    
            if current_count > last_count:
                start_time = time.time() 
                total_count = total_count + current_count - last_count 
            if current_count < last_count:
                duration = int(time.time() - start_time)
                if duration < threshold_value:
                    total_count = total_count - 1  
                if duration >= 4:
                    ### Topic "person/duration": key of "duration" ###
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    ### Topic "person": keys of "count" and "total" ###
                    client.publish("person", json.dumps({"total": total_count})) 
            
            ### current_count, total_count and duration to the MQTT server ###
            client.publish("person", json.dumps({"count": count}))
            last_count = current_count
             
            if key_pressed == 27:
                break 

        ### TODO: Send the frame to the FFMPEG server ###   
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
            
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT 
    client.disconnect()   

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()