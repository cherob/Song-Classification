const tf = require("@tensorflow/tfjs-node-gpu");
// require('@tensorflow/tfjs-node');
const Config = require('./cfg').Config;
const cliProgress = require('cli-progress');
const util = require('./util.js');

class Trainer {
  /**
   * @param {Config} config Config file
   */
  constructor(config) {
    this.model = tf.sequential();
    this.config = config;
    this.checkpoint = undefined;
  }

  generateModelReccurent(shape) {

    // Build and compile this model
    this.model.add(tf.layers.lstm({
      units: 128,
      returnSequences: true,
      inputShape: shape
    }));
    this.model.add(tf.layers.lstm({
      units: 128,
      returnSequences: true
    }));

    this.model.add(tf.layers.dropout(0.5));

    this.model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: 64,
        activation: "relu"
      })
    }));
    this.model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: 32,
        activation: "relu"
      })
    }));
    this.model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: 16,
        activation: "relu"
      })
    }));
    this.model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: 8,
        activation: "relu"
      })
    }));

    this.model.add(tf.layers.flatten());

    this.model.add(
      tf.layers.dense({
        units: this.config.categories,
        activation: "softmax"
      })
    );
    this.model.summary();
    this.compile();
  }

  generateModelConv(shape) {
    // Build and compile  this.model.
    this.model.add(tf.layers.conv2d({
      filters: 16,
      kernelSize: (3, 3),
      padding: "same",
      inputShape: shape,
      activation: "relu",
      strides: (1, 1)
    }));
    this.model.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: (3, 3),
      padding: "same",
      activation: "relu",
      strides: (1, 1)
    }));
    this.model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: (3, 3),
      padding: "same",
      activation: "relu",
      strides: (1, 1)
    }));
    this.model.add(tf.layers.conv2d({
      filters: 128,
      kernelSize: (3, 3),
      padding: "same",
      activation: "relu",
      strides: (1, 1)
    }));
    this.model.add(tf.layers.conv2d({
      filters: 256,
      kernelSize: (3, 3),
      padding: "same",
      activation: "relu",
      strides: (1, 1)
    }));
    this.model.add(tf.layers.conv2d({
      filters: 512,
      kernelSize: (3, 3),
      padding: "same",
      activation: "relu",
      strides: (1, 1)
    }));

    this.model.add(tf.layers.maxPool2d([2, 2]));
    this.model.add(tf.layers.dropout(0.5));
    this.model.add(tf.layers.flatten());

    this.model.add(tf.layers.dense({
      units: 512,
      activation: "relu"
    }));

    this.model.add(tf.layers.dense({
      units: 256,
      activation: "relu"
    }));

    this.model.add(tf.layers.dense({
      units: 128,
      activation: "relu"
    }));
    this.model.add(tf.layers.dense({
      units: 64,
      activation: "relu"
    }));
    this.model.add(
      tf.layers.dense({
        units: this.config.categories,
        activation: "softmax"
      })
    );

    this.model.summary();
    this.compile();
  }

  compile() {
    this.model.compile({
      loss: tf.metrics.categoricalCrossentropy,
      optimizer: tf.train.adam(this.config.learning_rate),
      metrics: ["acc"]
    });
  }

  async summary(X, y) {
    X = tf.tensor(X);
    let acc_list = [];
    let predictions = await this.model.predict(X).array();
    predictions.forEach((prediction, i) => {
      util.fromCategorical(y)[i].forEach((cat, ii) => {
        if (cat == 1)
          acc_list.push(prediction[util.fromCategorical(y)[i][ii]]);
      });
    });
    let acc = ((util.average(acc_list)) * 100).toFixed(2);
    let loss = (100 - util.standardDeviation(acc_list) * 100).toFixed(2);
    console.log(` -> [Validation] Acc: ${acc}% | Loss: ${loss}% `);
  }

  async load() {
    try {
      this.model = await tf.loadLayersModel(this.config.model_path + '/model.json');
      this.compile();
      return true;
    } catch (error) {
      return false;
    }
  }

  async start(dataset) {
    if (this.config.use_checkpoints_model)
      await this.load();

    this.config.acc_temp = [];
    this.config.val_acc_temp = [];
    this.config.loss_temp = [];
    this.config.val_loss_temp = [];

    if (!this.config.calls) this.config.calls = Number.MAX_VALUE;

    // Print config settings
    util.calculate(this.config);

    for (let i = 1; i < this.config.calls + 1; i++) {
      let X, y, vel_X, vel_y;
      let shuffled_model = [null, null];
      let history = {
        loss: [],
        acc: [],
        epoch: [0, this.config.epochs],
      }

      // Test Model
      vel_X = dataset.data.test.X;
      vel_y = util.toCategorical(dataset.data.test.y, this.config.categories);
      await this.summary(vel_X, vel_y);

      // Convert model
      X = dataset.data.train.X
      y = util.toCategorical(dataset.data.train.y, this.config.categories)

      // Shuffle model
      if (this.config.shuffle) {
        console.log(" -> Shuffle model");
        shuffled_model = util.shuffle(X, y);
      }

      // Convert to tensor
      X = tf.tensor(shuffled_model[0] || X);
      y = tf.tensor(shuffled_model[1] || y);

      // Loading Bar
      let progress = [0, 0];
      let bar_options = [{
        format: 'Epoch {epoch} [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} | Acc: {acc}% | Loss: {loss}% '
      }, cliProgress.Presets.shades_classic];
      var bar2 = new cliProgress.SingleBar(...bar_options);


      // Start fitting
      let fit_options = {
        verbose: 0,
        epochs: this.config.epochs,
        shuffle: this.config.shuffle_fit,
        batch_size: this.config.batch_size,
        callbacks: {
          onYield: function (epochs, batch, logs) {
            util.printOnce(`=== Call ${i} === `);

            // Set current Epoch
            history.epoch[0] = epochs + 1;

            if (bar2.value == 0)
              bar2.start(dataset.data.train.X.length, 0, {
                acc: (util.average(history.acc.slice(-progress[1])) * 100).toFixed(2),
                loss: (100 - util.standardDeviation(history.acc.slice(-progress[1])) * 100).toFixed(2),
                epoch: `${history.epoch[0]}/${history.epoch[1]}`,
                dot: new Array(batch % 1 + batch % 2 + batch % 3).fill('.').join('')
              });

          },
          onBatchEnd: function (batch, logs) {
            progress[0] += logs.size;
            progress[1]++;

            history.acc.push(logs.acc)
            history.loss.push(logs.loss)

            bar2.update(progress[0], {
              acc: (util.average(history.acc.slice(-progress[1])) * 100).toFixed(2),
              loss: (100 - util.standardDeviation(history.acc.slice(-progress[1])) * 100).toFixed(2),
              epoch: `${history.epoch[0]}/${history.epoch[1]}`,
              dot: new Array(batch % 1 + batch % 2 + batch % 3).fill('.').join('')
            });
          },
          onEpochEnd: () => {
            bar2.stop();
            bar2 = new cliProgress.SingleBar(...bar_options);
            progress[0] = 0;
          }
        }
      }

      this.checkpoint = await this.model.fit(X, y, fit_options);


      await this.model.save(this.config.model_path);
    }
  }
}

module.exports = {
  Trainer: Trainer
};