# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import datetime
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
#import tensorflow as tf2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import resource

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None: 
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):        
        tf.disable_v2_behavior()
        tf_config= tf.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5))
        
        self.session = tf.Session(config=tf_config)
        print("Loading protobuf file...")
        t1 = datetime.datetime.now()
        # with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(file_handle.read())
        
        # First deserialize your frozen graph:
        print("Loading protobuf file...")
        with tf.gfile.GFile("resources/networks/mars-small128.pb", "rb") as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        print("Loading complete...")
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        # print("Converting...")
        # trt_graph = trt.create_inference_graph(
        #     input_graph_def=frozen_graph,
        #     outputs=["features"],
        #     max_batch_size=1,
        #     max_workspace_size_bytes=1 << 26,
        #     precision_mode='FP16',
        #     minimum_segment_size=2
        # )
        tf.import_graph_def(frozen_graph, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        #self.input_var.set_shape([None, 128])
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)
        #self.output_var.set_shape([None, 128, 64, 3])

        #assert len(self.output_var.get_shape()) == 2
        #assert len(self.input_var.get_shape()) == 4
        #self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.feature_dim = 128
        #self.image_shape = self.input_var.get_shape().as_list()[1:]
        self.image_shape = [128,64,3]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def show_graph_info(graph):
    #count how many ops in trt_graph
    trt_engine_ops = len([1 for n in graph.node if str(n.op)=='TRTEngineOp'])
    print("numb. of trt_engine_ops in trt_graph", trt_engine_ops)
    all_ops = len([1 for n in graph.node])
    print("numb. of all_ops in in trt_graph:", all_ops)


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def load_model_from_metacheck():
    with tf.Session() as s:
        loader = tf.train.import_meta_graph("resources/networks/mars-small128.ckpt-68577.meta")
        loader.restore(s, "resources/networks/mars-small128.ckpt-68577")
        
        builder = tf.saved_model.builder.SavedModelBuilder("resources/networks/mars-small128_savedmodel/")
        builder.save()


def convert_to_trt():
    print("Loading SavedModel...")
    #conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode='FP16',
        is_dynamic_op=True)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir="resources/networks/mars-small128_savedmodel/",
        conversion_params=conversion_params)
    # Converter method used to partition and optimize TensorRT compatible segments
    print("Converting to TF-TRT...")
    converter.convert()
    print("Saving...")
    converter.save("resources/networks/mars-small128_tf-trt")


def convert_to_trt_from_frozen():
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        print("Loading protobuf file...")
        with tf.gfile.GFile("resources/networks/mars-small128.pb", "rb") as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        print("Loading complete...")
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        print("Converting...")
        # converter = trt.TrtGraphConverter(
        #     input_graph_def=frozen_graph,
        #     nodes_denylist=["features"]) #output nodes
        # converter = trt.TrtGraphConverter(
        #     input_saved_model_dir="resources/networks/mars-small128_savedmodel/",
        #     input_saved_model_tags=["images"],
        #     nodes_denylist=["features"]) #output nodes
        # trt_graph = converter.convert()
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=["features"],
            max_batch_size=1,
            max_workspace_size_bytes=1 << 26,
            precision_mode='FP16',
            minimum_segment_size=2
        )
        print("Conversion done.")
        print("Saving...")
        with open("resources/networks/mars-small128_trt", 'wb') as f:
            f.write(trt_graph.SerializeToString())
                
        #converter.save("resources/networks/mars-small128_trt/")
        print("Saved.")

def get_graph():
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        print("Loading protobuf file...")
        with tf.gfile.GFile("resources/networks/mars-small128.pb", "rb") as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        print("Loading complete...")
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        print("Converting...")
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=["features"],
            max_batch_size=32,
            max_workspace_size_bytes=1 << 26,
            precision_mode='FP16',
            minimum_segment_size=2,
            is_dynamic_op=True
        )
        return trt_graph
        

def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print("Processing {} - Mem: {} MB".format(sequence, usage))
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int("".join(filter(str.isdigit, os.path.splitext(f)[0]))): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        t1 = datetime.datetime.now()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print("Frame {:05d}/{:05d} - Mem: {}".format(frame_idx, max_frame_idx, usage/1024))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]
        t2 = datetime.datetime.now()
        print("feature extraction time: {} seconds".format((t2-t1).seconds))

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    #load_model_from_metacheck()
    #convert_to_trt()
    #convert_to_trt_from_frozen()
    
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
