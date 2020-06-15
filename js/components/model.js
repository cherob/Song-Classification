const tf = require("@tensorflow/tfjs-node-gpu");

module.exports = {
    reccurent: function (inputShape, outputShape) {
        let model = tf.sequential();

        // Build and compile this model
        model.add(tf.layers.lstm({
            units: 128,
            returnSequences: true,
            inputShape: inputShape
        }));
        model.add(tf.layers.lstm({
            units: 128,
            returnSequences: true
        }));

        model.add(tf.layers.dropout(0.5));

        model.add(tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units: 64,
                activation: "relu"
            })
        }));
        model.add(tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units: 32,
                activation: "relu"
            })
        }));
        model.add(tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units: 16,
                activation: "relu"
            })
        }));
        model.add(tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units: 8,
                activation: "relu"
            })
        }));

        model.add(tf.layers.flatten());

        model.add(
            tf.layers.dense({
                units: outputShape,
                activation: "softmax"
            })
        );

        return model;
    },

    conventional: function (inputShape, outputShape) {
        let model = tf.sequential();

        // Build and compile  model.
        model.add(tf.layers.conv2d({
            filters: 512,
            kernelSize: (3, 3),
            padding: "same",
            inputShape: inputShape,
            activation: "relu",
            strides: (1, 1)
        }));
        model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: (3, 3),
            padding: "same",
            activation: "relu",
            strides: (1, 1)
        }));
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: (3, 3),
            padding: "same",
            activation: "relu",
            strides: (1, 1)
        }));
        model.add(tf.layers.conv2d({
            filters: 128,
            kernelSize: (3, 3),
            padding: "same",
            activation: "relu",
            strides: (1, 1)
        }));
        model.add(tf.layers.conv2d({
            filters: 256,
            kernelSize: (3, 3),
            padding: "same",
            activation: "relu",
            strides: (1, 1)
        }));
        model.add(tf.layers.maxPool2d([2, 2]));
        model.add(tf.layers.dropout(0.5));
        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: 256,
            activation: "relu"
        }));

        model.add(tf.layers.dense({
            units: 128,
            activation: "relu"
        }));
        model.add(tf.layers.dense({
            units: 64,
            activation: "relu"
        }));
        model.add(
            tf.layers.dense({
                units: outputShape.length,
                activation: "softmax"
            })
        );

        return model;
    }
};