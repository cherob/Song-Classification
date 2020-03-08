const tf = require("@tensorflow/tfjs-node-gpu");
// require('@tensorflow/tfjs-node');
const Config = require('./cfg').Config;
const cliProgress = require('cli-progress');
const util = require('./util/util.js');

class Trainer {
  /**
   * @param {Config} config Config file
   */
  constructor(config) {
    this.model = tf.sequential();
    this.config = config;
    this.checkpoint = undefined;
  }

  drawStatistics() {
    // let high = this.config.acc_temp.length - 1
    // console.log(this.config.acc_temp[high],
    //   this.config.acc_temp[high],
    //   this.config.loss_temp[high],
    //   this.config.val_loss_temp[high]);
  }

  generateModel(shape) {
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

    this.model.compile({
      optimizer: "adam",
      loss: "meanSquaredError",
      metrics: ["acc"]
    });
  }

  async start(dataset) {
    this.config.acc_temp = [];
    this.config.val_acc_temp = [];
    this.config.loss_temp = [];
    this.config.val_loss_temp = [];

    if (!this.config.calls) this.config.calls = Number.MAX_VALUE;

    for (let i = 1; i < this.config.calls + 1; i++) {
      this.drawStatistics();

      let X, y, vel_X, vel_y;
      X = dataset.data.train.X.tolist()
      y = util.to_categorical(dataset.data.train.y.tolist(), this.config.categories)

      vel_X = dataset.data.test.X.tolist();
      vel_y = util.to_categorical(dataset.data.test.y.tolist(), this.config.categories);

      let shuffled_model = [null, null];
      if (this.config.shuffle) {
        console.log(" -> Shuffle model");
        shuffled_model = util.shuffle(X, y);
      }
      X = tf.tensor(shuffled_model[0] || X);
      y = tf.tensor(shuffled_model[1] || y);

      let bar_options = [{
        format: 'Epoch {epoch} [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} | Acc: {acc} ({std_dev}, {real_acc}) | Loss: {loss} '
      }, cliProgress.Presets.shades_classic];

      let averages = {
        loss: [],
        acc: [],
        epoch: [0, this.config.epochs],
      }

      let progress = 0;
      var bar2 = new cliProgress.SingleBar(...bar_options);

      // start fitting
      this.checkpoint = await this.model.fit(X, y, {
        epochs: this.config.epochs,
        shuffle: this.config.shuffle_fit,
        batch_size: this.config.batch_size,
        verbose: 0,
        validationData: tf.tensor([vel_X, vel_y]),
        callbacks: {
          onEpochBegin: function (epochs, logs) {},
          onTrainBegin: function (logs) {},
          onTrainEnd: function (logs) {

          },
          onBatchBegin: function (batch, logs) {},
          onYield: function (epochs, batch, logs) {
            util.printOnce(`=== Call ${i} === `);
            averages.epoch[0] = epochs + 1;

            if (bar2.value == 0) {
              bar2.start(dataset.data.train.X.tolist().length, 0, {
                acc: 0.000,
                loss: 1.000,
                std_dev: 0.00,
                dot: 0.00,
                epoch: `${averages.epoch[0]}/${averages.epoch[1]}`,
              });
            }
          },
          onBatchEnd: function (batch, logs) {
            progress += logs.size;
            averages.acc.push(logs.acc)
            averages.loss.push(logs.loss)

            bar2.update(progress, {
              acc: average(averages.acc).toFixed(3),
              loss: logs.loss.toFixed(3),
              std_dev: util.standardDeviation(averages.acc).toFixed(2),
              real_acc: logs.acc.toFixed(2),
              epoch: `${averages.epoch[0]}/${averages.epoch[1]}`,
              dot: new Array(batch % 1 + batch % 2 + batch % 3).fill('.').join('')
            });
          },
          onEpochEnd: function (epochs, logs) {
            bar2.stop();
            progress = 0;
            averages = {
              loss: [],
              acc: [],
              epoch: [0, averages.epoch[1]]
            }
            bar2 = new cliProgress.SingleBar(...bar_options);
          }
        }
      });

      this.model.save(this.config.model_path);
    }
  }
}

module.exports = {
  Trainer: Trainer
};