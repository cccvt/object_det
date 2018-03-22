TensorFlow saving into/loading a graph from a file
https://stackoverflow.com/questions/38947658/tensorflow-saving-into-loading-a-graph-from-a-file

There are many ways to approach the problem of saving a model in TensorFlow, which can make it a bit confusing. Taking each of your sub-questions in turn:

The checkpoint files (produced e.g. by calling saver.save() on a tf.train.Saver object) contain only the weights, and any other variables defined in the same program. To use them in another program, you must re-create the associated graph structure (e.g. by running code to build it again, or calling tf.import_graph_def()), which tells TensorFlow what to do with those weights. Note that calling saver.save() also produces a file containing a MetaGraphDef, which contains a graph and details of how to associate the weights from a checkpoint with that graph. See the tutorial for more details.
tf.train.write_graph() only writes the graph structure; not the weights.
Bazel is unrelated to reading or writing TensorFlow graphs. (Perhaps I misunderstand your question: feel free to clarify it in a comment.)
A frozen graph can be loaded using tf.import_graph_def(). In this case, the weights are (typically) embedded in the graph, so you don't need to load a separate checkpoint.
The main change would be to update the names of the tensor(s) that are fed into the model, and the names of the tensor(s) that are fetched from the model. In the TensorFlow Android demo, this would correspond to the inputName and outputName strings that are passed to TensorFlowClassifier.initializeTensorFlow().
The GraphDef is the program structure, which typically does not change through the training process. The checkpoint is a snapshot of the state of a training process, which typically changes at every step of the training process. As a result, TensorFlow uses different storage formats for these types of data, and the low-level API provides different ways to save and load them. Higher-level libraries, such as the MetaGraphDef libraries, Keras, and skflow build on these mechanisms to provide more convenient ways to save and restore an entire model.
