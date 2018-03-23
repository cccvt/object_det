import tensorflow as tf
g = tf.Graph()

class ImportGraph():
    """  Importing and running isolated TF graph """

    def __init__(self, loc):
        # Create local graph and use it in the session
        # self.graph = tf.Graph()
        self.graph = g
        # self.sess = tf.Session(graph=self.graph)
        with g.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(loc, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

            self.sess = tf.Session(graph=g)

    # def run(self, param, data):
    #     """ Running the activation operation previously imported """
    #     # The 'x' corresponds to name of input placeholder
    #     return self.sess.run(self.activation, feed_dict={param: data})