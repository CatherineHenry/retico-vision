from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')


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


    def set_image(self, image, nframes, rate):
        """Sets the audio content of the IU."""
        self.image = image
        self.payload = image
        self.nframes = int(nframes)
        self.rate = int(rate)


    def get_json(self):
        payload = {}
        payload['image'] = np.array(self.payload).tolist()
        payload['nframes'] = self.nframes
        payload['rate'] = self.rate
        return payload

    def create_from_json(self, json_dict):
        self.image =  Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.payload = self.image
        self.nframes = json_dict['nframes']
        self.rate = json_dict['rate']


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


    def set_detected_objects(self, image, detected_objects, object_type):
        """Sets the content for the IU"""
        self.image = image
        self.payload = detected_objects
        self.detected_objects = detected_objects
        self.num_objects = len(detected_objects)
        self.object_type = object_type


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
        self.payload = None
        self.num_objects = 0
        self.image = None
        self.image_bbox = None


    def set_payload(self, image, object_features, image_bbox):
        """Sets the content of the IU."""
        self.image = image
        self.payload = object_features
        self.num_objects = len(object_features)
        self.image_bbox = image_bbox

    def get_json(self):
        payload = {}
        # print(type(self.object_features))
        payload['image'] = np.array(self.image).tolist()
        payload['payload'] = self.payload
        payload['image_bbox'] = self.image_bbox
        payload['num_objects'] = self.num_objects
        return payload

    def create_from_json(self, json_dict):
        self.image = Image.fromarray(np.array(json_dict['image'], dtype='uint8'))
        self.image_bbox = json_dict['image_bbox']
        self.payload = json_dict['payload']
        self.num_objects = json_dict['num_objects']


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
            output_iu.meta_data= {'execution_uuid': 'test', 'flow_uuid': 'test'}
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
        return "A module that produces images of individual objects from segmentations produced by SAM or Yolo."

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
                image_position_feats = {}
                image_bbox = {}
                output_iu = self.create_iu(iu)
                execution_uuid = iu.meta_data.get('execution_uuid')
                flow_uuid = iu.meta_data.get('flow_uuid')
                date_timestamp = iu.meta_data.get('date_timestamp')
                print(f"Extracting objects [{flow_uuid}]")

                image = iu.image
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                image = image.convert('RGBA')
                num_objs = iu.num_objects
                obj_type = iu.object_type
                # print(f"Num Objects in Vision: {num_objs}")

                num_obj_to_display = self.num_obj_to_display
                if (num_obj_to_display > num_objs):
                    num_obj_to_display = num_objs
                    print(f"Number of objects detected less than requested [{num_objs} detected]. Showing {num_obj_to_display} objects.")


                input_image = np.array(image) #need image to be in numpy.ndarray format for methods
                if obj_type == 'bb':
                    valid_boxes = iu.detected_objects
                    for i in range(num_objs):
                        res_image_array = self.extract_bb_object(input_image, valid_boxes[i])
                        if res_image_array is None:
                            continue
                        res_image = Image.fromarray(res_image_array)
                        if self.show:
                            res_image.show()

                        position_feats = self.compute_position_feats(image, valid_boxes[i])
                        image_objects[f'object_{i+1}'] = res_image
                        image_position_feats[f'object_{i+1}'] = position_feats
                        x1, y1, x2, y2 = [int(val) for val in valid_boxes[i]]
                        image_bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'input_img_h': image.height, 'input_img_w': image.width}
                    output_iu.set_extracted_objects(image, image_objects, num_objs, obj_type, image_position_feats, image_bbox)
                elif obj_type == 'seg':
                    valid_segs = iu.detected_objects
                    for i in range(num_objs):
                        extracted = self.extract_seg_object(input_image, valid_segs[i])
                        if extracted is None:
                            continue
                        else:
                            res_image = Image.fromarray(extracted).convert('RGB')
                        image_objects[f'object_{i+1}'] = res_image # would run faster if I quit the loop here
                    output_iu.set_extracted_objects(image, image_objects, num_objs, obj_type)
                else: 
                    print('Object type is invalid. Can\'t retrieve segmented object.')
                    exit()
                # if all(value is None for value in image_objects.values()):
                if len(image_objects.keys()) == 0:
                    print(f"No images with object [{flow_uuid}]")
                    output_iu.set_extracted_objects(image, [], 0, obj_type)
                    # um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                    # self.append(um)
                # print(image_objects)
                else:
                    path = Path(f"{self.base_filepath}/{date_timestamp}/{obj_type}/{execution_uuid}/extracted/")
                    path.mkdir(parents=True, exist_ok=True)
                    file_name = f"{flow_uuid}.png" # TODO: png or jpg better?
                    imwrite_path = f"{str(path)}/{file_name}"
                    try:
                        res_image.save(imwrite_path)
                    except FileNotFoundError:
                        print(f"Did not write extracted image output to {imwrite_path}. Check directory exists.")

                # else: # TODO: Catherine: The plotting all works but isn't necessary atm
                #     plt.clf()
                #     num_rows = math.ceil(num_obj_to_display / 3)
                #     if num_obj_to_display < 3:
                #         num_cols = num_obj_to_display
                #     else:
                #         num_cols = 3
                #     fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows)) #need to adjust to have matching columsn and rows to fit num_obj_to_display
                #     axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
                #     od = collections.OrderedDict(sorted(image_objects.items(), reverse=True))
                #
                #     for idx, i in enumerate(od.keys()):
                #         if idx > num_obj_to_display: break
                #         res_image = od[i]
                #         if res_image is None:
                #             continue
                #         axs[idx-1].imshow(res_image)
                #         axs[idx-1].set_title(i)
                #
                #     for j in range(num_obj_to_display, num_rows * num_cols):
                #         axs[j].axis('off')
                #
                #     plt.tight_layout()
                #     path = Path(f"{self.base_filepath}/{obj_type}/{iu.execution_uuid}/top_n_extracted/")
                #     path.mkdir(parents=True, exist_ok=True)
                #     file_name = f"{iu.flow_uuid}.png" # TODO: png or jpg better?
                #     imwrite_path = f"{str(path)}/{file_name}"
                #     plt.savefig(imwrite_path)
                #     plt.close('all')
                #
                #
                #     # Print possible objects that could have been saved
                #     plt.clf()
                #     num_rows = math.ceil(len(image_objects.keys()) / 3)
                #     if len(image_objects.keys()) < 3:
                #         num_cols = len(image_objects.keys())
                #     else:
                #         num_cols = 3
                #     fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows)) #need to adjust to have matching columsn and rows to fit num_obj_to_display
                #     axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
                #
                #     for idx, i in enumerate(image_objects.keys()):
                #         res_image = image_objects[i]
                #         if res_image is None:
                #             continue
                #         axs[idx-1].imshow(res_image)
                #         axs[idx-1].set_title(i)
                #
                #     for j in range(num_objs, num_rows * num_cols):
                #         axs[j].axis('off')
                #
                #     plt.tight_layout()
                #     path = Path(f"{self.base_filepath}/{obj_type}/{iu.execution_uuid}/all_possible/")
                #     path.mkdir(parents=True, exist_ok=True)
                #     file_name = f"{iu.flow_uuid}.png" # TODO: png or jpg better?
                #     imwrite_path = f"{str(path)}/{file_name}"
                #     plt.savefig(imwrite_path)
                #     plt.close('all')

            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD) 
            self.append(um)

    def compute_position_feats(self, og_img, bbox):

        bbox = [int(val) for val in bbox]
        ih = og_img.height
        iw = og_img.width
        x, y, w, h = bbox
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        x,y,w,h = bbox
        # x1, relative
        x1r = x / iw
        # y1, relative
        y1r = y / ih
        # x2, relative
        x2r = (x+w) / iw
        # y2, relative
        y2r = (y+h) / ih
        # area
        area = (w*h) / (iw*ih)
        # ratio image sides (= orientation)
        ratio = iw / ih
        # distance from center (normalised)
        cx = iw / 2
        cy = ih / 2
        bcx = x + w / 2
        bcy = y + h / 2
        distance = np.sqrt((bcx-cx)**2 + (bcy-cy)**2) / np.sqrt(cx**2+cy**2)
        # done!
        return np.array([x1r,y1r,x2r,y2r,area,ratio,distance]).reshape(1,7)

    def extract_seg_object(self, image, seg):
        ret_image = image.copy()
        ret_image[seg==False] = [255,255,255,0]
        avg_color_per_row = np.average(ret_image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_avg = np.average(avg_color, axis=0)
        print(f"avg of seg obj is: {avg_avg}")
        # if avg_avg >= 200: #TODO: figure out a good threshold
        if avg_avg >= 190: # this is if the entire image is masked
            return None
        # ret_image[seg==True] = [255, 255, 255]
        return ret_image
    
    def extract_bb_object(self, image, bbox):
        # Note: Masked and cropped both return np.ndarray of image

        #return a cut out of the bounding boxed object from the image 
        if not self.keepmask:
            # Yolov8 returns bbox as left, top, right, bottom
            # casting to int does result in some data loss (potentially smaller or bigger bounding box)
            x1, y1, x2, y2 = [int(val) for val in bbox]
            # ret_image = image.copy()
            # ret_image[seg==True] = [255,255,255,0]
            ret_image = image[y1:y2, x1:x2]
        else: # Does not crop the image, rather keeps original image and whites out the area of the mask
            # keep position of object in image
            mask = np.zeros_like(image)
            x1, y1, x2, y2 = [int(val) for val in bbox] # cast to ints to circumvent issue with cv2 rect

            cv2.rectangle(mask, (0, 0), (image.shape[1], image.shape[0]), (255, 255, 255), -1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            mask = cv2.bitwise_not(mask)
            
            ret_image = cv2.bitwise_and(image, image, mask=mask)

            ret_image[mask == 0] = [255, 255, 255]
        # Catherine: Changes from working with SAM segments (trying to drop whitespace images)
        # # ret_image = cv2.cvtColor(ret_image, cv2.COLOR_RGB2BGR)
        # avg_color_per_row = np.average(ret_image, axis=0)
        # avg_color = np.average(avg_color_per_row, axis=0)
        # avg_avg = np.average(avg_color, axis=0)
        # print(f"avg of seg obj is: {avg_avg}")
        # # if avg_avg >= 200: #TODO: figure out a good threshold
        # if avg_avg >= 191.25: # this is if the entire image is masked
        #     return None
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
        self.image_bbox = None


    def set_extracted_objects(self, image, objects_dictionary, num_objects, object_type, image_position_feats=None, image_bbox=None):
        """Sets the content for the IU"""
        self.image = image
        self.payload = objects_dictionary
        self.num_objects = num_objects
        self.object_type = object_type
        self.extracted_objects = objects_dictionary
        self.image_position_feats = image_position_feats if image_position_feats is not None else {}
        self.image_bbox = image_bbox if image_bbox is not None else {}

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
                

class ObjectPermanenceIU(retico_core.IncrementalUnit):
    """An object permanence incremental unit that maintains a list of feature vectors for detected objects in a scene.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """

    @staticmethod
    def type():
        return "Object Permanence IU"

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
        self.payload = None


    def set_payload(self, distance_mm, obj_x1, obj_y1, obj_x2, obj_y2, orig_img_width):
        """Sets the content of the IU."""
        self.payload = {'distance_mm': distance_mm,
                        'obj_x1': obj_x1,
                        'obj_y1': obj_y1,
                        'obj_x2': obj_x2,
                        'obj_y2': obj_y2,
                        'orig_img_width': orig_img_width
                        }



class CozmoNavigationMemoryMapIU(retico_core.IncrementalUnit):
    """A cozmo navigation memory map incremental unit that maintains the latest navigation memory map state from the Cozmo engine.

    Attributes:
        creator (AbstractModule): The module that created this IU
        previous_iu (IncrementalUnit): A link to the IU created before the
            current one.
        grounded_in (IncrementalUnit): A link to the IU this IU is based on.
        created_at (float): The UNIX timestamp of the moment the IU is created.
    """

    @staticmethod
    def type():
        return "Cozmo Navigation Memory Map IU"

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
        self.payload = None


    def set_payload(self, nav_memory_map):
        """Sets the content of the IU."""
        self.payload = nav_memory_map
