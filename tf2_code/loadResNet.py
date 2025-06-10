import argparse
import tensorflow as tf
import keras2onnx


def keras_to_pb(model, output_filename, output_node_names):
 
    """
    This is the function to convert the keras model to pb.
 
    Args:
       model: The keras model.
       output_filename: The output .pb file name.
       output_node_names: The output nodes of the network (if None, 
       the function gets the last layer name as the output node).
    """
    sess = tf.compat.v1.keras.backend.get_session()
    graph = sess.graph

    with graph.as_default():
        # Get names of input and output nodes.
        in_name = model.layers[0].get_output_at(0).name.split(':')[0]
    
        if output_node_names is None:
            output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]
    
        graph_def = graph.as_graph_def()
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            output_node_names)

    sess.close()
    wkdir = ''
    tf.compat.v1.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)
 
    return in_name, output_node_names


def main(args):
    # Disable eager execution in tensorflow 2 is required.
    tf.compat.v1.disable_eager_execution()
    # Set learning phase to Test.
    tf.compat.v1.keras.backend.set_learning_phase(0)

    # load ResNet50 model pre-trained on imagenet
    model = tf.keras.applications.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000
    )
     
    # Convert keras ResNet50 model to .pb file
    in_tensor_name, out_tensor_names = keras_to_pb(model, args.output_pb_file , None) 

    # # You can also use keras2onnx
    # onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=11)
    # keras2onnx.save_model(onnx_model, "resnet.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_pb_file', type=str, default='resnet50.pb')
    args=parser.parse_args()
    main(args)

