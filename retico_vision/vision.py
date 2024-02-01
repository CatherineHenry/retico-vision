from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

import retico_core



class ImageIU(retico_core.IncrementalUnit):
    """An image incremental unit that receives raw image data from a source.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
        image (bytes[]): The image of this IU
        rate (int): The frame rate of this IU
        nframes (int): The number of frames of this IU
    """

    @staticmethod
    def type():
        return "Image IU"

    def __init__(
        self, 
        creator=None, 
        iuid=0, 
        previous_iu=None, 
        grounded_in=None,
        rate=None,
        nframes=None, 
        image=None,
        **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=image
        )
        self.image = image
        self.rate = rate
        self.nframes = nframes
        self.flow_uuid = None
        self.execution_uuid = None
        self.motor_action = None

    def set_image(self, image, nframes, rate):
        """Sets the audio content of the IU."""
        self.image = image
        self.payload = image
        self.nframes = int(nframes)
        self.rate = int(rate)

    def set_flow_uuid(self, flow_uuid):
        self.flow_uuid = flow_uuid

    def set_execution_uuid(self, execution_uuid):
        self.execution_uuid = execution_uuid

    def set_motor_action(self, motor_action):
        self.motor_action = motor_action

    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist()
        payload['nframes'] = self.nframes
        payload['rate'] = self.rate
        payload['execution_uuid'] = self.execution_uuid
        payload['flow_uuid'] = self.flow_uuid
        payload['motor_action'] = self.motor_action.tolist()
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.payload = self.image
        self.nframes = json_dict['nframes']
        self.rate = json_dict['rate']
        self.flow_uuid = json_dict['flow_uuid']
        self.execution_uuid = json_dict['execution_uuid']
        self.motor_action = np.array(json_dict['motor_action'])

class DetectedObjectsIU(retico_core.IncrementalUnit):
    """An image incremental unit that maintains a list of detected objects and their bounding boxes.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """

    @staticmethod
    def type():
        return "Detected Objects IU"

    def __init__(
        self, 
        creator=None, 
        iuid=0, 
        previous_iu=None,
        grounded_in=None,
        **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid, 
            previous_iu=previous_iu,
            grounded_in=grounded_in, 
            payload=None
        )
        self.image = None
        self.detected_objects = None
        self.num_objects = 0
        self.object_type = None
        self.flow_uuid = None
        self.execution_uuid = None
        self.motor_action = None

    def set_detected_objects(self, image, detected_objects, object_type):
        """Sets the content for the IU"""
        self.image = image
        self.payload = detected_objects
        self.detected_objects = detected_objects
        self.num_objects = len(detected_objects)
        self.object_type = object_type

    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist() #just sets image to be the list of objects 
        payload['detected_objects'] = self.detected_objects
        payload['num_objects'] = self.num_objects
        payload['object_type'] = self.object_type
        payload['flow_uuid'] = self.flow_uuid
        payload['execution_uuid'] = self.execution_uuid
        payload['motor_action'] = self.motor_action.tolist()
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.detected_objects = json_dict['detected_objects']
        self.payload = self.detected_objects
        self.num_objects = json_dict['num_objects']
        self.flow_uuid = json_dict['flow_uuid']
        self.execution_uuid = json_dict['execution_uuid']
        self.motor_action = np.array(json_dict['motor_action'])

    def set_flow_uuid(self, flow_uuid):
        self.flow_uuid = flow_uuid

    def set_execution_uuid(self, execution_uuid):
        self.execution_uuid = execution_uuid

    def set_motor_action(self, motor_action):
        self.motor_action = motor_action

class ObjectFeaturesIU(retico_core.IncrementalUnit):
    """An image incremental unit that maintains a list of feature vectors for detected objects in a scene.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """

    @staticmethod
    def type():
        return "Object Features IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=None
        )
        self.object_features = None
        self.num_objects = 0
        self.image = None
        self.flow_uuid = None
        self.execution_uuid = None
        self.motor_action = None

    def set_flow_uuid(self, flow_uuid):
        self.flow_uuid = flow_uuid

    def set_execution_uuid(self, execution_uuid):
        self.execution_uuid = execution_uuid

    def set_motor_action(self, motor_action):
        self.motor_action = motor_action

    def set_object_features(self, image, object_features):
        """Sets the content of the IU."""
        self.image = image
        self.payload = object_features
        self.object_features = object_features
        self.num_objects = len(object_features)

    def get_json(self):
        payload = {}
        # print(type(self.object_features))
        payload['image'] = np.array(self.image).tolist()
        payload['object_features'] = self.object_features
        payload['num_objects'] = self.num_objects
        payload['flow_uuid'] = self.flow_uuid
        payload['motor_action'] = self.motor_action.tolist()
        payload['execution_uuid'] = self.execution_uuid
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.object_features = json_dict['object_features']
        self.payload = json_dict['object_features']
        self.num_objects = json_dict['num_objects']
        self.flow_uuid = json_dict['flow_uuid']
        self.motor_action = np.array(json_dict['motor_action'])
        self.execution_uuid = json_dict['execution_uuid']

class WebcamModule(retico_core.AbstractProducingModule):
    """A module that produces IUs containing images that are captures by
    a web camera."""

    @staticmethod
    def name():
        return "Webcam Module"

    @staticmethod
    def description():
        return "A prodicing module that records images from a web camera."

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, width=None, height=None, rate=None, pil=True, **kwargs):
        """
        Initialize the Webcam Module.
        Args:
            width (int): Width of the image captured by the webcam; will use camera default if unset
            height (int): Height of the image captured by the webcam; will use camera default if unset
            rate (int): The frame rate of the recording; will use camera default if unset
        """
        super().__init__(**kwargs)
        self.pil = pil
        self.width = width
        self.height = height
        self.rate = rate
        self.cap = cv2.VideoCapture(0)

        self.setup()

    def process_update(self, _):
        ret, frame = self.cap.read() # ret should be false if camera is off
        if ret:
            output_iu = self.create_iu()
            # output_iu.set_image(frame, self.width, self.height, self.rate)
            if self.pil:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame2 = np.asarray(frame)
                cv2.imwrite("./test_webcam_img.jpg", frame2)
            output_iu.set_image(frame, 1, self.rate)
            return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            # um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            # self.append(um)
        else:
            print('camera may not be on')

    def setup(self):
        """Set up the webcam for recording."""
        cap = self.cap
        if self.width != None:
            cap.set(3, self.width)
        else:
            self.width = int(cap.get(3))
        if self.height != None:
            cap.set(4, self.height)
        else:
            self.height = int(cap.get(4))
        if self.rate != None:
            cap.set(5, self.rate)
        else:
            self.rate = int(cap.get(5))

    def shutdown(self):
        """Close the video stream."""
        self.cap.release()    
        
    
class ImageCropperModule(retico_core.AbstractModule):
    """A module that crops images"""

    @staticmethod
    def name():
        return "Image Cropper Module"

    @staticmethod
    def description():
        return "A module that crops images"


    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, top=-1, bottom=-1, left=-1, right=-1, **kwargs):
        """
        Initialize the Webcam Module.
        Args:
            width (int): Width of the image captured by the webcam; will use camera default if unset
            height (int): Height of the image captured by the webcam; will use camera default if unset
            rate (int): The frame rate of the recording; will use camera default if unset
        """
        super().__init__(**kwargs)
        self.top =  top
        self.bottom = bottom
        self.left = left
        self.right = right

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            image = iu.image
            width, height = image.size
            left = self.left if self.left != -1 else 0
            top = self.top if self.top != -1 else 0
            right = self.right if self.right != -1 else width
            bottom = self.bottom if self.bottom != -1 else height
            image = image.crop((left, top, right, bottom)) 
            output_iu = self.create_iu(iu)
            output_iu.set_image(image, iu.nframes, iu.rate)
            return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        
        return None
        
        
class ExtractObjectsModule(retico_core.AbstractModule):
    """A module that produces image IUs containing detected objects segmented 
    by SAM or Yolo."""

    @staticmethod
    def name():
        return "Extract Object Module"

    @staticmethod
    def description():
        return "A module that produces iamges of individual objects from segmentations produced by SAM or Yolo."

    @staticmethod
    def input_ius():
        return [DetectedObjectsIU]
    
    @staticmethod
    def output_iu():
        return ExtractedObjectsIU

    def __init__(self, num_obj_to_display=1, show=False, keepmask=False, **kwargs):
        """
        Initialize the Display Objects Module
        Args:
            object_type (str): whether object is defined 
                in bounding box or segmentation
            num_obj_to_display (int): amount of objects from
                detected objects to display 
        """
        super().__init__(**kwargs)
        self.num_obj_to_display = num_obj_to_display
        self.show = show
        self.keepmask = keepmask
        self.base_filepath = './extraction_output'

    # TODO: Catherine, no queue for this Module?
    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                image_objects = {}
                output_iu = self.create_iu(iu)
                print(f"Extracting objects [{iu.flow_uuid}]")

                img_dict = iu.get_json()
                image = iu.image
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                image = image.convert('RGBA')
                obj_type = img_dict['object_type']
                num_objs = img_dict['num_objects']
                # print(f"Num Objects in Vsison: {num_objs}")

                if (self.num_obj_to_display > num_objs):
                    print("Number of objects detected less than requested. Showing all objects.")
                    self.num_obj_to_display = num_objs

                sam_image = np.array(image) #need image to be in numpy.ndarray format for methods
                if obj_type == 'bb':
                    valid_boxes = img_dict['detected_objects']
                    for i in range(num_objs):
                        res_image = self.extract_bb_object(sam_image, valid_boxes[i])
                        if self.show:
                            cv2.imshow('image',res_image) 
                            cv2.waitKey(1)
                        image_objects[f'object_{i+1}'] = res_image
                    output_iu.set_extracted_objects(image, image_objects, num_objs, obj_type)
                elif obj_type == 'seg':
                    valid_segs = img_dict['detected_objects']
                    for i in range(num_objs):
                        extracted = self.extract_seg_object(sam_image, valid_segs[i])
                        if extracted is None:
                            continue
                        else:
                            res_image = Image.fromarray(extracted).convert('RGB')
                        image_objects[f'object_{i+1}'] = res_image
                    output_iu.set_extracted_objects(image, image_objects, num_objs, obj_type)
                else: 
                    print('Object type is invalid. Can\'t retrieve segmented object.')
                    exit()
                # if all(value is None for value in image_objects.values()):
                if len(image_objects.keys()) == 0:
                    print(f"No images with object [{iu.flow_uuid}]")
                    output_iu.set_extracted_objects(image, [], 0, obj_type)
                    # um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                    # self.append(um)
                # print(image_objects)
                else:
                    plt.clf()
                    num_rows = math.ceil(self.num_obj_to_display / 3)
                    if self.num_obj_to_display < 3:
                        num_cols = self.num_obj_to_display
                    else:
                        num_cols = 3
                    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows)) #need to adjust to have matching columsn and rows to fit num_obj_to_display
                    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

                    for idx, i in enumerate(image_objects.keys()):
                        res_image = image_objects[i]
                        if res_image is None:
                            continue
                        axs[idx].imshow(res_image)
                        axs[idx].set_title(i)

                    for j in range(self.num_obj_to_display, num_rows * num_cols):
                        axs[j].axis('off')

                    plt.tight_layout()
                    path = Path(f"{self.base_filepath}/{obj_type}/{iu.execution_uuid}/top_n_extracted/")
                    path.mkdir(parents=True, exist_ok=True)
                    file_name = f"{iu.flow_uuid}.png" # TODO: png or jpg better?
                    imwrite_path = f"{str(path)}/{file_name}"
                    plt.savefig(imwrite_path)
                    plt.close('all')


                    # Print possible objects that could have been saved
                    plt.clf()
                    num_rows = math.ceil(num_objs / 3)
                    if num_objs < 3:
                        num_cols = num_objs
                    else:
                        num_cols = 3
                    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows)) #need to adjust to have matching columsn and rows to fit num_obj_to_display
                    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

                    for idx, i in enumerate(image_objects.keys()):
                        res_image = image_objects[i]
                        if res_image is None:
                            continue
                        axs[idx].imshow(res_image)
                        axs[idx].set_title(i)

                    for j in range(num_objs, num_rows * num_cols):
                        axs[j].axis('off')

                    plt.tight_layout()
                    path = Path(f"{self.base_filepath}/{obj_type}/{iu.execution_uuid}/all_possible/")
                    path.mkdir(parents=True, exist_ok=True)
                    file_name = f"{iu.flow_uuid}.png" # TODO: png or jpg better?
                    imwrite_path = f"{str(path)}/{file_name}"
                    plt.savefig(imwrite_path)
                    plt.close('all')

            output_iu.set_flow_uuid(iu.flow_uuid)
            output_iu.set_execution_uuid(iu.execution_uuid)
            output_iu.set_motor_action(iu.motor_action)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD) 
            self.append(um)

    def extract_seg_object(self, image, seg):
        ret_image = image.copy()
        ret_image[seg==True] = [255,255,255,0]
        avg_color_per_row = np.average(ret_image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_avg = np.average(avg_color, axis=0)
        print(f"avg of seg obj is: {avg_avg}")
        if avg_avg >= 200: #TODO: figure out a good threshold
            return None
        # ret_image[seg==True] = [255, 255, 255]
        return ret_image
    
    def extract_bb_object(self, image, bbox):
        #return a cut out of the bounding boxed object from the image 
        if not self.keepmask:
            x, y, w, h = bbox
            ret_image = image[y:y+h, x:x+w]
        else:
            # keep position of object in image
            mask = np.zeros_like(image)
            x, y, w, h = bbox

            cv2.rectangle(mask, (0, 0), (image.shape[1], image.shape[0]), (255, 255, 255), -1)
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 0), -1)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            mask = cv2.bitwise_not(mask)
            
            ret_image = cv2.bitwise_and(image, image, mask=mask)

            ret_image[mask == 0] = [255, 255, 255]

        ret_image = cv2.cvtColor(ret_image, cv2.COLOR_RGB2BGR)

        return ret_image

        

class ExtractedObjectsIU(retico_core.IncrementalUnit):
    """A dictionary incremental unit that maintains a dictionary of objects segmented from an Image
    
    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the c
            current one
        grounded_in (IncrementalUnit): A link to the IU this IU is based on
        created_at (float): The UNIX timestamp of the moment the IU is created
    """

    @staticmethod
    def type():
        return "Extracted Objects IU"
    
    def __init__(
            self,
            creator=None,
            iuid=0,
            previous_iu=None,
            grounded_in=None,
            **kwargs
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=None
        )
        self.image = None
        self.num_objects = 0
        self.object_type = None
        self.extracted_objects = {}
        self.flow_uuid = None
        self.execution_uuid = None
        self.motor_action = None

    def set_flow_uuid(self, flow_uuid):
        self.flow_uuid = flow_uuid

    def set_execution_uuid(self, execution_uuid):
        self.execution_uuid = execution_uuid

    def set_motor_action(self, motor_action):
        self.motor_action = motor_action

    def set_extracted_objects(self, image, objects_dictionary, num_objects, object_type,):
        """Sets the content for the IU"""
        self.image = image
        self.payload = objects_dictionary
        self.num_objects = num_objects
        self.object_type = object_type
        self.extracted_objects = objects_dictionary

    def get_json(self):
        payload = {}
        payload['image'] = self.image
        payload['num_objects'] = self.num_objects
        payload['object_type'] = self.object_type
        payload['segmented_objects_dictionary'] = self.extracted_objects
        payload['flow_uuid'] = self.flow_uuid
        payload['motor_action'] = self.motor_action.tolist()
        payload['execution_uuid'] = self.execution_uuid
        return payload
    
    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.num_objects = json_dict['num_objects']
        self.extracted_objects = json_dict['segmented_objects_dictionary']
        self.payload = self.extracted_objects
        self.flow_uuid = json_dict['flow_uuid']
        self.motor_action = np.array(json_dict['motor_action'])
        self.execution_uuid = json_dict['execution_uuid']
                

