# coding=utf-8
import blobconverter
import cv2
import depthai as dai
import numpy as np
from MultiMsgSync import TwoStageHostSeqSync
from depthai_sdk.visualize.bbox import BoundingBox
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
from time import sleep  # Import the sleep function from the time module

GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(11, GPIO.OUT, initial=GPIO.LOW)  # Set pin 18 to be an output pin and set initial value to low (off)
GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)  # Set pin 18 to be an output pin and set initial value to low (off)
GPIO.setup(16, GPIO.OUT, initial=GPIO.LOW)  # Set pin 18 to be an output pin and set initial value to low (off)
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)  # Set pin 18 to be an output pin and set initial value to low (off)

VIDEO_SIZE = (1072, 1072)

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
openvino_version = '2021.4'


def create_output(name: str, output: dai.Node.Output):
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(name)
    output.link(xout.input)


cam = pipeline.create(dai.node.ColorCamera)
# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(1072, 1072)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewNumFramesPool(20)
cam.setFps(20)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setImageOrientation(dai.CameraImageOrientation.HORIZONTAL_MIRROR)
create_output('color', cam.video)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
face_det_manip.setMaxOutputFrameSize(300 * 300 * 3)
cam.preview.link(face_det_manip.inputImage)

# =================[ FACE DETECTION ]=================

print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(
    name="face-detection-retail-0004",
    shaves=6,
    version=openvino_version
))
# Link Face ImageManip -> Face detection NN node
face_det_manip.out.link(face_det_nn.input)

create_output('detection', face_det_nn.out)

# =================[ SCRIPT NODE ]=================

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
face_det_nn.passthrough.link(script.inputs['face_pass'])

cam.preview.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

# =================[ HEAD POSE ESTIMATION ]=================

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
script.outputs['headpose_cfg'].link(headpose_manip.inputConfig)
script.outputs['headpose_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(
    name="head-pose-estimation-adas-0001",
    shaves=6,
    version=openvino_version
))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

# =================[ LANDMARKS DETECTION ]=================

landmark_manip = pipeline.create(dai.node.ImageManip)
landmark_manip.initialConfig.setResize(48, 48)
script.outputs['landmark_cfg'].link(landmark_manip.inputConfig)
script.outputs['landmark_img'].link(landmark_manip.inputImage)

landmark_nn = pipeline.create(dai.node.NeuralNetwork)
landmark_nn.setBlobPath(blobconverter.from_zoo(
    name="landmarks-regression-retail-0009",
    shaves=6,
    version=openvino_version
))
landmark_manip.out.link(landmark_nn.input)

landmark_nn.out.link(script.inputs['landmark_in'])
landmark_nn.passthrough.link(script.inputs['landmark_pass'])

create_output('landmarks', landmark_nn.out)

# =================[ LEFT EYE CROP ]=================

left_manip = pipeline.create(dai.node.ImageManip)
left_manip.initialConfig.setResize(60, 60)
left_manip.inputConfig.setWaitForMessage(True)
script.outputs['left_manip_img'].link(left_manip.inputImage)
script.outputs['left_manip_cfg'].link(left_manip.inputConfig)
left_manip.out.link(script.inputs['left_eye_in'])

# =================[ LEFT EYE CROP ]=================

right_manip = pipeline.create(dai.node.ImageManip)
right_manip.initialConfig.setResize(60, 60)
right_manip.inputConfig.setWaitForMessage(True)
script.outputs['right_manip_img'].link(right_manip.inputImage)
script.outputs['right_manip_cfg'].link(right_manip.inputConfig)
right_manip.out.link(script.inputs['right_eye_in'])

# =================[ GAZE ESTIMATION ]=================

gaze_nn = pipeline.create(dai.node.NeuralNetwork)
gaze_nn.setBlobPath(blobconverter.from_zoo(
    name="gaze-estimation-adas-0002",
    shaves=6,
    version=openvino_version,
    compile_params=['-iop head_pose_angles:FP16,right_eye_image:U8,left_eye_image:U8']
))

SCRIPT_OUTPUT_NAMES = ['to_gaze_head', 'to_gaze_left', 'to_gaze_right']
NN_NAMES = ['head_pose_angles', 'left_eye_image', 'right_eye_image']
for script_name, nn_name in zip(SCRIPT_OUTPUT_NAMES, NN_NAMES):
    # Link Script node output to NN input
    script.outputs[script_name].link(gaze_nn.inputs[nn_name])
    # Set NN input to blocking and to not reuse previous msgs
    gaze_nn.inputs[nn_name].setBlocking(True)
    gaze_nn.inputs[nn_name].setReusePreviousMessage(False)

# Workaround, so NNData (output of gaze_nn) will take seq_num from this message (FW bug)
# Will be fixed in depthai 2.24
gaze_nn.passthroughs['left_eye_image'].link(script.inputs['none'])
script.inputs['none'].setBlocking(False)
script.inputs['none'].setQueueSize(1)

create_output('gaze', gaze_nn.out)

# ==================================================

with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
    try:
        sync = TwoStageHostSeqSync()

        queues = {}
        # Create output queues
        for name in ["color", "detection", "landmarks", "gaze"]:
            queues[name] = device.getOutputQueue(name)

        while True:
            for name, q in queues.items():
                # Add all msgs (color frames, detections and gaze estimations) to the Sync class.
                if q.has():
                    msg = q.get()
                    sync.add_msg(msg, name)
                    # if name == "color":
                    # cv2.imshow("video", msg.getCvFrame())

            msgs = sync.get_msgs()
            if msgs is not None:
                frame = msgs["color"].getCvFrame()
                dets = msgs["detection"].detections
                for i, detection in enumerate(dets):
                    try:
                        det = BoundingBox(detection)
                        tl, br = det.denormalize(frame.shape)
                        cv2.rectangle(frame, tl, br, (10, 245, 10), 1)

                        gaze = np.array(msgs["gaze"][i].getFirstLayerFp16())
                        gaze_x, gaze_y = (gaze * 100).astype(int)[:2]

                        landmarks = np.array(msgs["landmarks"][i].getFirstLayerFp16())
                        colors = [(0, 127, 255), (0, 127, 255), (255, 0, 127), (127, 255, 0), (127, 255, 0)]
                        for lm_i in range(0, len(landmarks) // 2):
                            GPIO.output(13, GPIO.LOW)  # Turn off
                            GPIO.output(11, GPIO.LOW)  # Turn off
                            GPIO.output(16, GPIO.LOW)  # Turn off
                            GPIO.output(18, GPIO.LOW)  # Turn off
                            # 0,1 - left eye, 2,3 - right eye, 4,5 - nose tip, 6,7 - left mouth, 8,9 - right mouth
                            x, y = landmarks[lm_i * 2:lm_i * 2 + 2]
                            point = det.map_point(x, y).denormalize(frame.shape)
                            arrowX = point[0] + gaze_x * 5
                            arrowY = point[1] - gaze_y * 5
                            # determine direction
                            direction = ""
                            TRESHOLD = 10
                            if gaze_x > x - 0.3:
                                if abs(gaze_y - y) <= TRESHOLD:
                                    direction = 'right'
                                    GPIO.output(13, GPIO.HIGH)
                                    GPIO.output(11, GPIO.HIGH) # Turn on

                                elif gaze_y > y:
                                    direction = 'right-up'
                                    GPIO.output(13, GPIO.HIGH)
                                    GPIO.output(11, GPIO.HIGH)  # Turn on
                                elif gaze_y < y:
                                    direction = 'right-down'
                                    GPIO.output(13, GPIO.HIGH)
                                    GPIO.output(11, GPIO.HIGH) # Turn on
                            elif gaze_x < x - 0.3:
                                if abs(gaze_y - y) <= TRESHOLD:
                                    direction = 'left'
                                    GPIO.output(16, GPIO.HIGH)  # Turn on
                                    GPIO.output(18, GPIO.HIGH)

                                elif gaze_y > y:
                                    direction = 'left-up'
                                    GPIO.output(16, GPIO.HIGH)  # Turn on
                                    GPIO.output(18, GPIO.HIGH)
                                elif gaze_y < y:
                                    direction = 'left-down'
                                    GPIO.output(16, GPIO.HIGH)  # Turn on
                                    GPIO.output(18, GPIO.HIGH)
                            elif gaze_x == x:
                                if gaze_y == y:
                                    direction = 'forward'
                                elif gaze_y > y:
                                    direction = 'up'
                                elif gaze_y < y:
                                    direction = 'down'

                            if lm_i <= 1:  # Draw arrows from left eye & right eye
                                cv2.arrowedLine(frame, point, (arrowX, arrowY), colors[lm_i], 3)
                                cv2.putText(frame, f'Person is looking {direction}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 0, 0), 2)
                            cv2.circle(frame, point, 2, colors[lm_i], 2)
                    except:
                        continue
            cv2.imshow("Lasers", frame)
            if cv2.waitKey(1) == ord('q'):
                GPIO.cleanup()
                break
    except KeyboardInterrupt:
        # Cleanup GPIO on Ctrl+C
        GPIO.cleanup()



