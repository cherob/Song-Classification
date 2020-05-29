const tf = require("@tensorflow/tfjs-node-gpu");
// require('@tensorflow/tfjs-node');
const Config = require('./cfg.v2').Config;
const stats = require('./analytics');
const fs = require(`fs`);
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

    this.improvement = false;
    this.last_score = 0;

  }

  generateModelReccurent(shape) {
    console.log(` -> Mode: Recurrent [${this.config.modes[this.config.mode]}]`) +

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

  generateModelNewVersion(shape) {
    console.log(` -> Mode: New [${this.config.modes[this.config.mode]}]`)
    this.model.add(tf.layers.flatten({
      inputShape: shape
    }))

    this.model.add(tf.layers.dense({
      units: 512,
      activation: "relu"
    }));

    this.model.add(tf.layers.dense({
      units: 256,
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

  generateModelConv(shape) {
    console.log(` -> Mode: Conv [${this.config.modes[this.config.mode]}]`)

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
    let optimizer = tf.train.adam(this.config.learning_rate);

    this.model.compile({
      loss: tf.metrics.categoricalCrossentropy,
      optimizer: optimizer,
      metrics: ["accuracy"]
    });
  }

  async summary(X, y) {

    let accuracies = [];

    this.improvement = false;

    try {
      let res = this.model.predict(tf.tensor(X));
      let predictions = res.arraySync();

      predictions.forEach((prediction, i) => {
        util.fromCategorical(y)[i].forEach((cat, ii) => {
          if (cat == 1)
            accuracies.push(prediction[util.fromCategorical(y)[i][ii]]);
        });
      });

    } catch (error) {
      console.error(error);
    }

    let acc = parseFloat(((util.average(accuracies)) * 100).toFixed(2));
    let loss = (100 - util.standardDeviation(accuracies) * 100).toFixed(2);

    if (acc > this.last_score) {
      this.last_score = acc;
      this.improvement = true;
    }

    console.log(` -> [Validation] Loss: ${loss}% | Acc: ${acc}% ${this.improvement ? ' => better!' : ''}`);
  }

  async start(dataset) {
    let result_params = {
      total_batch_size: 0,
      epoch: 0,
      logs: [],
      history: function () {
        let acc = this.logs.map((log) => log.acc);
        let loss = this.logs.map((log) => log.loss);

        return {
          epoch: this.epoch,
          acc: parseFloat(((util.average(acc.slice(Math.max(acc.length - (10), 0)))) * 100).toFixed(2)) || 0,
          loss: parseFloat(((util.average(loss.slice(Math.max(loss.length - (10), 0)))) * 100).toFixed(2)) || 0
        }
      },
      progress: function (size) {
        this.total_batch_size += size;
        return this.total_batch_size;
      }
    };

    if (!this.config.calls) this.config.calls = Number.MAX_VALUE;

    // Print config settings
    util.calculate(this.config);

    console.log('=== Save Images ===')
    stats.saveImages(this.config, dataset.data.train.X);


    for (let i = 1; i < this.config.calls + 1; i++) {
      try {
        if (this.config.use_checkpoints_model) {
          this.model = await tf.loadLayersModel(this.config.model_path + '/model.json');
          let params = fs.readFileSync(this.config.history_path);
          params = JSON.parse(params);
          result_params.epoch = params.epoch;
          result_params.logs = params.logs;
        }
      } catch (error) {}

      this.compile();

      result_params.logs = [];
      let X, y, X_val, y_val;

      // Train model
      X = dataset.data.train.X
      y = util.toCategorical(dataset.data.train.y, this.config.categories)

      // Validation Model
      X_val = dataset.data.test.X;
      y_val = util.toCategorical(dataset.data.test.y, this.config.categories);

      // Shuffle model
      // let shuffled_model = [null, null];
      // if (this.config.shuffle) {
      //   // console.log(" -> Shuffle model");
      //   shuffled_model = util.shuffle(X, y);
      //   X = shuffled_model[0];
      //   y = shuffled_model[1];
      // }

      // Loading Bar
      let status_epoch_bar_options = [{
        format: `Epoch {epoch} [{bar}] | Acc: {acc}% | Loss: {loss}%`
      }, cliProgress.Presets.shades_classic];
      let epoch_bar = new cliProgress.SingleBar(...status_epoch_bar_options);

      await this.model.fit(tf.tensor(X), tf.tensor(y), {
        verbose: 0,
        epochs: this.config.epochs,
        shuffle: true,
        batchSize: this.config.batch_size,
        callbacks: {
          onEpochBegin(epoch, logs) {
            result_params.epoch++;
            epoch_bar.start(X.length, 0, result_params.history());
          },
          onEpochEnd(epoch, logs) {
            epoch_bar.stop();
            result_params.total_batch_size = 0;
          },
          onBatchEnd(batch, logs) {
            result_params.logs.push(logs);
            epoch_bar.update(result_params.progress(logs.size), result_params.history())
          }
        }
      });

      await this.summary(X_val, y_val);
      try {
        if (this.improvement) {
          fs.writeFileSync(this.config.history_path, JSON.stringify(result_params));
          await this.model.save(this.config.model_path);
        }
      } catch (error) {
        console.log(error);
      }
    }
  }
}

module.exports = {
  Trainer: Trainer
};